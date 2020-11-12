# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer

import time

class TRTR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, decoder_query, bbox_head_hidden_dimension = None, aux_loss=False, weighted=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            decoder_query: the query size (one side) to feed into the decoder
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.transformer = transformer

        hidden_dim = transformer.d_model

        self.decoder_query = decoder_query
        self.class_embed = nn.Linear(hidden_dim * decoder_query * decoder_query, 2) # fully connected, class: has object or not

        if not bbox_head_hidden_dimension:
            bbox_head_hidden_dimension = hidden_dim
        self.bbox_embed = MLP(hidden_dim * decoder_query * decoder_query, bbox_head_hidden_dimension, 4, 3)
        #for param in self.bbox_embed.parameters():
        #    print(type(param), param.size(), param.requires_grad)

        #self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.input_projs = nn.ModuleList(nn.Conv2d(num_channels, hidden_dim, kernel_size=1) for num_channels in backbone.num_channels_list)

        self.weighted = weighted
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(backbone.num_channels_list)))
            self.loc_weight = nn.Parameter(torch.ones(len(backbone.num_channels_list)))

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

            bakui TODO:
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x 2]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
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

            hs = hs.flatten(2) # for following fully connected layer
            # print("hs flattened : {}".format(hs.shape))
            hs_list.append(hs)


        # sum
        hs_cls = None
        hs_loc = None

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
            hs_cls, hs_loc =  weighted_avg(hs_list, cls_weight), weighted_avg(hs_list, loc_weight)
        else:
            hs_cls, hs_loc =  avg(hs_list), avg(hs_list)

        outputs_class = self.class_embed(hs_cls)
        start = time.time()
        outputs_coord = self.bbox_embed(hs_loc).sigmoid()  # 0 ~ 1
        #print("bbox regrasion head: {}".format(time.time() - start))
        out = {'pred_logit': outputs_class[-1], 'pred_box': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logit': a, 'pred_box': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

    def loss_labels(self, outputs, targets, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logit' in outputs
        src_logits = outputs['pred_logit'] # [bN, 2] (0: non-object, 1: object)

        target_classes = torch.as_tensor([t["label"].item() for t in targets], device=src_logits.device) # [bN]
        # print("target_classes: {}".format(target_classes))

        assert src_logits.ndim == 2
        loss_ce = F.cross_entropy(src_logits, target_classes) # TODO maybe we need to use different weight for object and non-object
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_box' in outputs

        all_src_boxes = outputs['pred_box'] # [bN, 4]
        all_target_boxes = torch.stack([t['bbox'] for t in targets]) # [bn, 4]
        # print("all_target_boxes: {}".format(all_target_boxes))
        assert all_src_boxes.shape == all_target_boxes.shape

        # only calculate the loss for bbox has the object
        idx = [id for id, t in enumerate(targets) if t["label"] == 1] # only extract the index with object
        # print("targets: {}".format(targets))
        src_boxes = all_src_boxes[idx]
        target_boxes = all_target_boxes[idx]
        # assert src_boxes.device == target_boxes.device

        # debug
        # print("loss boxes all target boxes: {}".format(all_target_boxes))
        # print("loss boxes all src boxes: {}".format(all_src_boxes))

        # print("loss boxes idx: {}".format(idx))
        # print("loss boxes selected target boxes: {}".format(target_boxes))
        # print("loss boxes selected src boxes: {}".format(src_boxes))

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def get_loss(self, loss, outputs, targets, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(t["label"].item() for t in targets) # 0: non-object, 1:  object
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
        """

        out_logits, out_bboxes = outputs['pred_logit'], outputs['pred_box']

        assert len(out_logits) ==  len(target_sizes)
        assert target_sizes.shape[1] == 2

        probs = F.softmax(out_logits, -1)
        scores, labels = probs.max(-1)

        # do not convert to [x0, y0, x1, y1] format
        # boxes = box_ops.box_cxcywh_to_xyxy(out_bboxes)
        boxes = out_bboxes.cpu()
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)

        #print("post process: img_h, img_w: {}, {}".format(img_h,img_w))
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct

        results = [{'score': s, 'label': l, 'box': b} for s, l, b in zip(scores, labels, boxes)]

        return results


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

    print("start build ")
    model = TRTR(
        backbone,
        transformer,
        decoder_query = args.decoder_query,
        aux_loss=args.aux_loss,
        weighted = args.weighted
    )

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    criterion = SetCriterion(weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
