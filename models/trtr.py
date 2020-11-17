# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       reg_l1_loss, neg_loss,
                       get_world_size, is_dist_avail_and_initialized)

from .backbone import build_backbone
from .transformer import build_transformer

import time
import numpy as np

class TRTR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, aux_loss=False, weighted=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.transformer = transformer

        hidden_dim = transformer.d_model

        # heatmap
        # TODO: try to use MLP or Conv2d_3x3 before fully-connected like CenterNet
        self.heatmap_embed = nn.Linear(hidden_dim, 1)
        self.heatmap_embed.bias.data.fill_(-2.19)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.input_projs = nn.ModuleList(nn.Conv2d(num_channels, hidden_dim, kernel_size=1) for num_channels in backbone.num_channels_list)

        self.weighted = weighted
        if self.weighted:
            self.hm_weight = nn.Parameter(torch.ones(len(backbone.num_channels_list)))
            self.bbox_weight = nn.Parameter(torch.ones(len(backbone.num_channels_list)))

        self.backbone = backbone
        self.aux_loss = aux_loss

        self.template_src_projs = []
        self.template_mask = None
        self.template_pos = None
        self.memory = []

    def forward(self, search_samples: NestedTensor, template_samples: NestedTensor = None):
        """ template_samples is a NestedTensor for template image:
               - samples.tensor: batched images, of shape [batch_size x 3 x H_template x W_template]
               - samples.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels
            search_samples is also a NestedTensor for searching image:
               - samples.tensor: batched images, of shape [batch_size x 3 x H_search x W_search]
               - samples.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_heatmap": The heatmap of the target bbox, of shape= [batch_size x (H_search x W_search) x 1]
               - "pred_dense_reg": The regression of bbox for all query (i.e. all pixels), of shape= [batch_size x (H_search x W_search) x 2]
                                   The regression reg O = [ p/stride - p_tilde], where p and p_tilde are the corrdinates
                                   in input and output, respectively.
               - "pred_dense_wh": The size of bbox for all query (i.e. all pixels), of shape= [batch_size x (H_search x W_search) x 2]
                                  The height and width values are normalized in [0, 1],
                                  relative to the size of each individual image (disregarding possible padding).
                                  See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(template_samples, (torch.Tensor)):
            template_samples = nested_tensor_from_tensor_list(template_samples)

        if isinstance(search_samples, (torch.Tensor)):
            search_samples = nested_tensor_from_tensor_list(search_samples)


        template_features = None
        template_pos = None
        if template_samples is not None:
            template_features, self.template_pos  = self.backbone(template_samples)

            self.template_mask = template_features[-1].mask
            self.template_src_projs = []
            for input_proj, template_feature in zip(self.input_projs, template_features):
                self.template_src_projs.append(input_proj(template_feature.tensors))

            self.memory = []

        start = time.time()
        search_features, search_pos  = self.backbone(search_samples)
        # print("search image feature extraction: {}".format(time.time() - start))
        search_mask = search_features[-1].mask

        assert search_mask is not None
        assert self.template_mask is not None

        search_src_projs = []
        for input_proj, search_feature in zip(self.input_projs, search_features):
            search_src_projs.append(input_proj(search_feature.tensors))


        hs_list = []
        for i, (template_src_proj, search_src_proj) in enumerate(zip(self.template_src_projs, search_src_projs)):
            if template_samples is not None:
                hs, memory = self.transformer(template_src_proj, self.template_mask, self.template_pos[-1], search_src_proj, search_mask, search_pos[-1])
                self.memory.append(memory)
            else:
                hs = self.transformer(template_src_proj, self.template_mask, self.template_pos[-1], search_src_proj, search_mask, search_pos[-1], self.memory[i])[0]

            hs_list.append(hs)


        # sum
        hs_hm = None
        hs_bbox = None

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)
            hs_hm, hs_bbox =  weighted_avg(hs_list, cls_weight), weighted_avg(hs_list, loc_weight)
            raise # debug
        else:
            hs_hm, hs_bbox =  avg(hs_list), avg(hs_list)

        outputs_heatmap = self.heatmap_embed(hs_hm) # we have different sigmoid process for training and inference, so we do not get sigmoid here.
        # print("size of outputs_heatmap: {}, ".format(outputs_heatmap.shape))

        outputs_bbox = self.bbox_embed(hs_bbox).sigmoid()
        # TODO: whether can you sigmoid() for the offset regression,
        # YoLo V3 uses sigmoid: https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
        outputs_bbox_reg = outputs_bbox.split(2,-1)[0]
        outputs_bbox_wh = outputs_bbox.split(2,-1)[1]
        # print("size of outputs_bbox_reg: {}, outputs_bbox_wh: {}".format(outputs_bbox_reg.shape, outputs_bbox_wh.shape))


        out = {'pred_heatmap': outputs_heatmap[-1], 'pred_bbox_reg': outputs_bbox_reg[-1], 'pred_bbox_wh': outputs_bbox_wh[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_heatmap, outputs_bbox_reg, outputs_bbox_wh)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_bbox_reg, outputs_bbox_wh):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_heatmap': a, 'pred_bbox_reg': b, 'pred_bbox_wh': c}
                for a, b, c  in zip(outputs_class[:-1], outputs_bbox_reg[:-1], outputs_bbox_wh[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for TRTR.
    """
    def __init__(self, weight_dict):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = [self.loss_heatmap, self.loss_bbox] # workaround for heatmap based boundbox

    def loss_heatmap(self, outputs, targets, num_boxes):
        """ Focal loss for the heatmap
        targets dicts must contain the key "heatmap" containing a tensor of dim [batch_size]
        """
        assert 'pred_heatmap' in outputs
        src_heatmap = torch.clamp(outputs['pred_heatmap'].sigmoid(), min=1e-4, max=1-1e-4) # [bN, output_height *  output_width, 1], clamp for focal loss
        target_heatmap = torch.stack([t['hm'] for t in targets]) # [bn, output_hegiht, output_width]
        target_heatmap = target_heatmap.flatten(1).unsqueeze(-1)  # [bn, output_hegiht *  output_width, 1]

        loss_hm = neg_loss(src_heatmap, target_heatmap)

        losses = {'loss_hm': loss_hm}

        return losses

    def loss_bbox(self, outputs, targets, num_boxes):
        """Compute the losses related to the bounding boxes regression, the L1 regression loss
           targets dicts must contain the key "boxes" containing a tensor of dim [batch_size, 2]
           The target boxes regression are expected in format (gt_x / stride - floor(gt_x / stride), gt_x / stride - floor(gt_x / stride)).
           The target boxes width/height are expected in format (w, h), which is normalized by the input size.
           TODO: check the effect with/ without the GIoU loss
        """
        assert 'pred_bbox_reg' in outputs and 'pred_bbox_wh' in outputs

        all_src_boxes_reg = outputs['pred_bbox_reg'] # [bN, output_hegiht * output_width, 2]
        all_target_boxes_reg = torch.stack([t['reg'] for t in targets]) # [bn, 2]
        all_src_boxes_wh = outputs['pred_bbox_wh'] # [bN, output_hegiht * output_width, 2]
        all_target_boxes_wh = torch.stack([t['wh'] for t in targets]) # [bn, 2]
        all_target_boxes_ind = torch.as_tensor([t['ind'].item() for t in targets], device = all_src_boxes_reg.device) # [bn]

        #print("all_target_boxes_reg: {}, all_src_boxes_reg: {}".format(all_target_boxes_reg.shape, all_src_boxes_reg.shape))
        #print("all_target_boxes_wh: {}, all_src_boxes_wh: {}".format(all_target_boxes_wh.shape, all_src_boxes_wh.shape))
        #print("all_target_boxes_ind: {}".format(all_target_boxes_ind.shape))


        # TODO: reservation for only calculate the loss for bbox has the object
        mask = [id for id, t in enumerate(targets) if t['valid'].item() == 1] # only extract the index with object
        src_boxes_reg = all_src_boxes_reg[mask]
        target_boxes_reg = all_target_boxes_reg[mask]
        src_boxes_wh = all_src_boxes_wh[mask]
        target_boxes_wh = all_target_boxes_wh[mask]
        target_boxes_ind = all_target_boxes_ind[mask]

        #loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox_reg = reg_l1_loss(src_boxes_reg, target_boxes_ind, target_boxes_reg)
        loss_bbox_wh = reg_l1_loss(src_boxes_wh, target_boxes_ind, target_boxes_wh)

        losses = {}
        losses['loss_bbox_reg'] = loss_bbox_reg.sum() / num_boxes
        losses['loss_bbox_wh'] = loss_bbox_wh.sum() / num_boxes

        return losses

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # TODO: this is a reserved function fro a negative sample training to improve the robustness like DasiamRPN
        num_boxes = sum(t['valid'].item() for t in targets)
        # print("num of valid boxes: {}".format(num_boxes)) # debug
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(loss(outputs, targets, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    l_dict = loss(aux_outputs, targets, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        return losses

class PostProcess(nn.Module):
    def __init__(self):

        super().__init__()

    @torch.no_grad()
    def forward(self, outputs):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
        """

        # do post sigmoid process
        heatmap = outputs['pred_heatmap'].sigmoid().squeeze(-1)

        out = {'pred_heatmap': heatmap, 'pred_bbox_reg': outputs['pred_bbox_reg'], 'pred_bbox_wh': outputs['pred_bbox_wh']}

        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    print("start build model")
    model = TRTR(
        backbone,
        transformer,
        aux_loss=args.aux_loss,
        weighted = args.weighted
    )

    weight_dict = {'loss_hm': 1, 'loss_bbox_reg': args.reg_loss_coef, 'loss_bbox_wh': args.wh_loss_coef}

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(weight_dict=weight_dict)
    criterion.to(device)

    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
