# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import datasets.utils
from datasets.utils import crop_image, siamfc_like_scale, get_exemplar_size # TODO: move to utils
from util.box_ops import box_cxcywh_to_xyxy
import cv2
import time

class Tracker(object):
    def __init__(self, model, postprocess, search_size):
        self.model = model
        self.model.eval()
        self.postprocess = postprocess

        self.search_size = search_size

        self.size_lpf = 0.33  # TODO: shift to args, (SiamRPN: 0.33),

        self.image_normalize = T.Compose([
            T.ToTensor(), # Converts a numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.first_frame = False

    def _bbox_clip(self, bbox, boundary):
        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(boundary[1], bbox[2])
        y2 = min(boundary[0], bbox[3])

        return [x1, y1, x2, y2]

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox (opencv format for rect)
        """

        bbox_xyxy  = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        self.center_pos = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        self.size = bbox[2:]
        channel_avg = np.mean(img, axis=(0, 1))

        # get crop
        scale_z = siamfc_like_scale(bbox_xyxy)[1]
        template_image, _ = crop_image(img, bbox_xyxy, padding = channel_avg)

        self.rect_template_image = template_image.copy()
        init_bbox = np.array(self.size) * scale_z
        exemplar_size = get_exemplar_size()
        x1 = np.round(exemplar_size/2 - init_bbox[0]/2).astype(np.uint8)
        y1 = np.round(exemplar_size/2 - init_bbox[1]/2).astype(np.uint8)
        x2 = np.round(exemplar_size/2 + init_bbox[0]/2).astype(np.uint8)
        y2 = np.round(exemplar_size/2 + init_bbox[1]/2).astype(np.uint8)
        cv2.rectangle(self.rect_template_image, (x1, y1), (x2, y2), (0,255,0), 3)

        # normalize and conver to torch.tensor
        self.template = self.image_normalize(np.round(template_image).astype(np.uint8)).unsqueeze(0).cuda()
        self.first_frame = True

        # debug
        return template_image

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        bbox_xyxy = [self.center_pos[0] - self.size[0] / 2,
                     self.center_pos[1] - self.size[1] / 2,
                     self.center_pos[0] + self.size[0] / 2,
                     self.center_pos[1] + self.size[1] / 2]
        channel_avg = np.mean(img, axis=(0, 1))
        _, search_image = crop_image(img, bbox_xyxy, padding = channel_avg, instance_size = self.search_size)

        # normalize and conver to torch.tensor
        search = self.image_normalize(np.round(search_image).astype(np.uint8)).unsqueeze(0).cuda()

        with torch.no_grad():
            if self.first_frame:
                outputs = self.model(search, self.template)
                self.first_frame = False
            else:
                outputs = self.model(search)

        outputs = self.postprocess(outputs, torch.as_tensor(search.shape[-2:]).unsqueeze(0))
        score = outputs[0]["score"]

        # print("outputs: {}".format(outputs))

        scale_z = siamfc_like_scale(bbox_xyxy)[1]

        ct_delta = (outputs[0]["bbox_ct"] - self.search_size / 2) / scale_z
        # print("scaled back bbox: {}, delta pos: {}, scaked back search size: {}".format(bbox, pos_delta, self.search_size / scale_z))
        cx = self.center_pos[0] + ct_delta[0].item()
        cy = self.center_pos[1] + ct_delta[1].item()

        # smooth bbox
        lpf = score * self.size_lpf
        bbox_wh = outputs[0]["bbox_wh"] / scale_z
        width = self.size[0] * (1 - lpf) + bbox_wh[0].item() * lpf
        height = self.size[1] * (1 - lpf) + bbox_wh[1].item() * lpf

        # clip boundary
        bbox = [cx - width / 2, cy - height / 2,
                cx + width / 2, cy + height / 2]
        bbox = self._bbox_clip(bbox, img.shape[:2])

        # udpate state
        self.center_pos = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # debug for search image:
        start_time = time.time()
        debug_bbox = box_cxcywh_to_xyxy(torch.round(torch.cat([outputs[0]["bbox_ct"], outputs[0]["bbox_wh"]])).int())
        rec_search_image = cv2.rectangle(search_image,
                                         (debug_bbox[0], debug_bbox[1]),
                                         (debug_bbox[2], debug_bbox[3]),(0,255,0),3)

        raw_heatmap = (torch.round(outputs[0]["raw_hm"] * 255)).detach().numpy().astype(np.uint8)
        post_heatmap = (torch.round(outputs[0]["post_hm"] * 255)).detach().numpy().astype(np.uint8)
        heatmap_resize = cv2.resize(raw_heatmap, search_image.shape[1::-1])
        heatmap_color = np.stack([heatmap_resize, np.zeros(search_image.shape[1::-1], dtype=np.uint8), heatmap_resize], -1)
        rec_search_image = np.round(0.4 * heatmap_color + 0.6 * rec_search_image.copy()).astype(np.uint8)
        # print("after postprocess time: {}".format(time.time() - start_time))

        return {
            'bbox': bbox,
            'score': score,
            'raw_heatmap': raw_heatmap,
            'post_heatmap': post_heatmap,
            'search_image': rec_search_image, # debug
            'template_image': self.rect_template_image # debug
               }

def build_tracker(model, postprocess, args):
    return Tracker(model, postprocess, args.search_size)
