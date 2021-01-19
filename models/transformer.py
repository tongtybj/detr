# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TRTR Transformer class.
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_


class SharedEncoderTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, encoder_feedforward = False, decoder_feedforward = True, activation="relu"):
        super().__init__()

        # shared self attention module
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # self-attention post norm
        self.self_attn_post_skip_connect = SkipConnection(d_model, dropout)

        # feedforward module after self attention
        self.encoder_feedforward = encoder_feedforward
        if self.encoder_feedforward:
            self.self_attn_post_feedforward = FeedForwardModule(d_model, dim_feedforward = dim_feedforward,
                                                                dropout = dropout, activation = activation)

        # encoder-decoder attention module
        self.decoder_attn = ModifiedMultiheadAttention(d_model, nhead, dropout=dropout)

        # encoder-decoder attention post norm
        self.decoder_attn_post_skip_connect = SkipConnection(d_model, dropout)

        # feedforward module after encoder-decoder attention
        self.decoder_feedforward = decoder_feedforward
        if self.decoder_feedforward:
            self.decoder_attn_post_feedforward = FeedForwardModule(d_model, dim_feedforward = dim_feedforward,
                                                                   dropout = dropout, activation = activation)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, template_src, template_mask, template_pos_embed, search_src, search_mask, search_pos_embed, memory = None):
        """
        template_src: [batch_size x hidden_dim x H_template x W_template]
        template_mask: [batch_size x H_template x W_template]
        template_pos_embed: [batch_size x hidden_dim x H_template x W_template]

        search_src: [batch_size x hidden_dim x H_search x W_search]
        search_mask: [batch_size x H_search x W_search]
        search_pos_embed: [batch_size x hidden_dim x H_search x W_search]
        """

        # flatten and permute bNxCxHxW to HWxbNxC for encoder in transformer
        template_src = template_src.flatten(2).permute(2, 0, 1)
        template_pos_embed = template_pos_embed.flatten(2).permute(2, 0, 1)
        if template_mask is not None:
            template_mask = template_mask.flatten(1)

        # encoding the template embedding with positional embbeding
        if memory is None:
            q = k = self.with_pos_embed(template_src, template_pos_embed)
            memory, attn_weight_map = self.self_attn(q, k, value=template_src,
                                                     key_padding_mask=template_mask)
            memory = self.self_attn_post_skip_connect(template_src, memory)
            if self.encoder_feedforward:
                memory = self.self_attn_post_feedforward(memory)


        # flatten and permute bNxCxHxW to HWxbNxC for decoder in transformer
        search_src = search_src.flatten(2).permute(2, 0, 1) # tgt
        search_pos_embed = search_pos_embed.flatten(2).permute(2, 0, 1)
        if search_mask is not None:
            search_mask = search_mask.flatten(1)

        q = k = self.with_pos_embed(search_src, search_pos_embed)
        search_query, attn_weight_map = self.self_attn(q, k, value=search_src,
                                                        key_padding_mask=search_mask)
        search_query = self.self_attn_post_skip_connect(search_src, search_query)
        if self.encoder_feedforward:
            search_query = self.self_attn_post_feedforward(search_query)

        tgt, attn_weight_map = self.decoder_attn(query=self.with_pos_embed(search_query, search_pos_embed),
                                                 key=self.with_pos_embed(memory, template_pos_embed),
                                                 value=memory,
                                                 key_padding_mask=template_mask)
        tgt = self.decoder_attn_post_skip_connect(search_query, tgt)
        if self.decoder_feedforward:
            tgt = self.decoder_attn_post_feedforward(tgt)

        return tgt.unsqueeze(0).transpose(1, 2), memory

class SkipConnection(nn.Module):

    def __init__(self, d_model, dropout = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src2):
        src = src + self.dropout(src2)
        src = self.norm(src)
        return src

class FeedForwardModule(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        self.skip_connect = SkipConnection(d_model, dropout)

    def forward(self, src):
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.skip_connect(src, src2)
        return src

class ModifiedMultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    Modified: give the same weight for key and query

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.):
        super(ModifiedMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"


        self.qk_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim)) # use the shared weight for query and key
        self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.register_parameter('in_proj_weight', None)

        self.in_qk_proj_bias = nn.Parameter(torch.empty(embed_dim))
        self.in_v_proj_bias = nn.Parameter(torch.empty(embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.bias_k = self.bias_v = None
        self.add_zero_attn = None

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.qk_proj_weight)
        xavier_uniform_(self.v_proj_weight)

        constant_(self.in_qk_proj_bias, 0.)
        constant_(self.in_v_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(ModifiedMultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):

        in_proj_bias = torch.cat([self.in_qk_proj_bias, self.in_qk_proj_bias, self.in_v_proj_bias])
        return F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.qk_proj_weight,
            k_proj_weight=self.qk_proj_weight,
            v_proj_weight=self.v_proj_weight)


class DefaultTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, template_src, template_mask, template_pos_embed, search_src, search_mask, search_pos_embed, memory = None):
        """
        template_src: [batch_size x hidden_dim x H_template x W_template]
        template_mask: [batch_size x H_template x W_template]
        template_pos_embed: [batch_size x hidden_dim x H_template x W_template]

        search_src: [batch_size x hidden_dim x H_search x W_search]
        search_mask: [batch_size x H_search x W_search]
        search_pos_embed: [batch_size x hidden_dim x H_search x W_search]
        """

        if len(template_src) > 1 and len(search_src) == 1:
            # print("do multiple frame mode ")
            template_src = template_src.flatten(2) # flatten: bNxCxHxW to bNxCxHW
            template_src = torch.cat(torch.split(template_src,1), -1) # concat: bNxCxHW to 1xCxbNHW
            template_src = template_src.permute(2, 0, 1) # permute 1xCxbNHW to bNHWx1xC for encoder in transformer

            template_pos_embed = template_pos_embed.flatten(2) # flatten: bNxCxHxW to bNxCxHW
            template_pos_embed = torch.cat(torch.split(template_pos_embed,1), -1) # concat: bNxCxHW to 1xCxbNHW
            template_pos_embed = template_pos_embed.permute(2, 0, 1) # permute 1xCxbNHW to bNHWx1xC for encoder in transformer

            if template_mask is not None:
                template_mask = template_mask.flatten(1) # flatten: bNxHxW to bNxHW
                template_mask = torch.cat(torch.split(template_mask,1), -1) # concat: bNxHW to 1xbNHW

        else:
            # flatten and permute bNxCxHxW to HWxbNxC for encoder in transformer
            template_src = template_src.flatten(2).permute(2, 0, 1)
            template_pos_embed = template_pos_embed.flatten(2).permute(2, 0, 1)
            if template_mask is not None:
                template_mask = template_mask.flatten(1)


        # encoding the template embedding with positional embbeding
        if memory is None:
            memory = self.encoder(template_src, src_key_padding_mask=template_mask, pos=template_pos_embed)

        # flatten and permute bNxCxHxW to HWxbNxC for decoder in transformer
        search_src = search_src.flatten(2).permute(2, 0, 1) # tgt
        search_pos_embed = search_pos_embed.flatten(2).permute(2, 0, 1)
        if template_mask is not None:
            search_mask = search_mask.flatten(1)

        hs = self.decoder(search_src, memory,
                          memory_key_padding_mask=template_mask,
                          tgt_key_padding_mask=search_mask,
                          encoder_pos=template_pos_embed,
                          decoder_pos=search_pos_embed)

        return hs.transpose(1, 2), memory


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                encoder_pos: Optional[Tensor] = None,
                decoder_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           encoder_pos=encoder_pos,
                           decoder_pos=decoder_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, attn_weight_map = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                               key_padding_mask=src_key_padding_mask)
        #print("encoder: self attn_weight_map: {}".format(attn_weight_map))
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     encoder_pos: Optional[Tensor] = None,
                     decoder_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, decoder_pos)
        tgt2, attn_weight_map = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                               key_padding_mask=tgt_key_padding_mask)
        #print("decoder: self attn_weight_map: {}".format(attn_weight_map))
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn_weight_map = self.multihead_attn(query=self.with_pos_embed(tgt, decoder_pos),
                                                          key=self.with_pos_embed(memory, encoder_pos),
                                                          value=memory, attn_mask=memory_mask,
                                                          key_padding_mask=memory_key_padding_mask)
        #print("decoder: multihead attn_weight_map: {}".format(attn_weight_map))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    encoder_pos: Optional[Tensor] = None,
                    decoder_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, decoder_pos)
        tgt2, attn_weight_map = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                               key_padding_mask=tgt_key_padding_mask)
        #print("decoder: self attn_weight_map: {}".format(attn_weight_map))
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn_weight_map = self.multihead_attn(query=self.with_pos_embed(tgt2, decoder_pos),
                                                    key=self.with_pos_embed(memory, encoder_pos),
                                                    value=memory, attn_mask=memory_mask,
                                                    key_padding_mask=memory_key_padding_mask)
        #print("decoder: multihead attn_weight_map: {}".format(attn_weight_map))
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                encoder_pos: Optional[Tensor] = None,
                decoder_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask,
                                    encoder_pos, decoder_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 encoder_pos, decoder_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):

    if args.use_default_transformer:
        return DefaultTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
    else:
        return SharedEncoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            encoder_feedforward = args.encoder_feedforward,
            decoder_feedforward = args.decoder_feedforward
        )



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
