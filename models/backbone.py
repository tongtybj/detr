# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone_name: str, backbone: nn.Module, train_backbone: bool, return_layers: List):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        return_layer_map = {}
        if not return_layers:
            return_layers = ["layer4"]
        for idx, layer in enumerate(return_layers):
            assert layer in ('layer2', 'layer3', 'layer4')
            return_layer_map[layer] = str(idx)
            #return_layer_map = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layer_map)

        self.num_channels_list = []
        for layer in return_layers:
            if layer == "layer2":
                if backbone_name == "resnet50":
                    self.num_channels_list.append(512)
            elif layer == "layer3":
                if backbone_name == "resnet50":
                    self.num_channels_list.append(1024)
            elif layer == "layer4":
                if backbone_name == "resnet50":
                    self.num_channels_list.append(2048)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_layers: List,
                 dilation: bool):
        if dilation:
            dilation = [False, True, True] # workaround to achieve stride of 8
        else:
            dilation = [False, False, False]

        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=dilation,
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        super().__init__(name, backbone, train_backbone, return_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

        self.num_channels_list = []

    def forward(self, tensor_list: NestedTensor):

        xs = self[0](tensor_list) # extract feature from search image (embedding)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

            # print("backbone {}: shape: {}".format(name, x.tensors.shape))

        return out, pos

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0

    backbone = Backbone(args.backbone, train_backbone, args.return_layers, args.resnet_dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels_list = backbone.num_channels_list
    return model
