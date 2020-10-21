# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from datasets.vid.curate_vid import crop_image, siamfc_like_scale # TODO: move to utils
from util.box_ops import box_cxcywh_to_xyxy
import cv2

class Tracker(object):
    def __init__(self, model, postprocess, exemplar_size, search_size, context_amount):
        self.model = model
        self.model.eval()
        self.postprocess = postprocess

        self.exemplar_size = exemplar_size
        self.search_size = search_size
        self.context_amount = context_amount

        self.size_lpf = 0.5

        self.image_normalize = T.Compose([
            T.ToTensor(), # Converts a numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox_center):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox (opencv format for rect)
        """

        bbox_xyxy  = [bbox_center[0], bbox_center[1],
                      bbox_center[0] + bbox_center[2],
                      bbox_center[1] + bbox_center[3]]
        self.center_pos = [bbox_center[0] + bbox_center[2]/2,
                           bbox_center[1] + bbox_center[3]/2]
        self.size = [bbox_center[2], bbox_center[3]]
        channel_avg = np.mean(img, axis=(0, 1))

        # get crop
        scale_z = siamfc_like_scale(bbox_xyxy, self.context_amount, self.exemplar_size)[1]
        template_image, _ = crop_image(img, bbox_xyxy, exemplar_size=self.exemplar_size, instance_size=self.search_size, padding = channel_avg, context_amount=self.context_amount)

        self.rect_template_image = template_image.copy()
        init_bbox = np.array(self.size) * scale_z
        x1 = np.round(self.exemplar_size/2 - init_bbox[0]/2).astype(np.uint8)
        y1 = np.round(self.exemplar_size/2 - init_bbox[1]/2).astype(np.uint8)
        x2 = np.round(self.exemplar_size/2 + init_bbox[0]/2).astype(np.uint8)
        y2 = np.round(self.exemplar_size/2 + init_bbox[1]/2).astype(np.uint8)
        cv2.rectangle(self.rect_template_image, (x1, y1), (x2, y2), (0,255,0), 3)

        # normalize and conver to torch.tensor
        self.template = self.image_normalize(np.round(template_image).astype(np.uint8)).unsqueeze(0).cuda()

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
        _, search_image = crop_image(img, bbox_xyxy, exemplar_size=self.exemplar_size, instance_size=self.search_size, padding = channel_avg, context_amount=self.context_amount)

        # normalize and conver to torch.tensor
        search = self.image_normalize(np.round(search_image).astype(np.uint8)).unsqueeze(0).cuda()

        with torch.no_grad():
            outputs = self.model(self.template, search)

        outputs = self.postprocess(outputs, torch.as_tensor(search.shape[-2:]).unsqueeze(0))

        print("outputs: {}".format(outputs))

        scale_z = siamfc_like_scale(bbox_xyxy, self.context_amount, self.exemplar_size)[1]

        bbox = outputs[0]["box"] / scale_z
        pos_delta = (outputs[0]["box"][:2] - self.search_size / 2) / scale_z

        print("scaled back bbox: {}, delta pos: {}, scaked back search size: {}".format(bbox, pos_delta, self.search_size / scale_z))

        cx = self.center_pos[0] + pos_delta[0]
        cy = self.center_pos[1] + pos_delta[1]

        # smooth bbox
        width = self.size[0] * (1 - self.size_lpf) + bbox[2].item() * self.size_lpf
        height = self.size[1] * (1 - self.size_lpf) + bbox[3].item() * self.size_lpf

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])



        # udpate state
        self.center_pos = [cx, cy]
        self.size = [width, height]

        bbox = [cx - width / 2, cy - height / 2,
                cx + width / 2, cy + height / 2]

        # debug for search image:
        debug_bbox = box_cxcywh_to_xyxy(torch.round(outputs[0]["box"]).int())
        rec_search_image = cv2.rectangle(search_image,
                                         (debug_bbox[0], debug_bbox[1]),
                                         (debug_bbox[2], debug_bbox[3]),(0,255,0),3)
        return {
            'bbox': bbox,
            'label': outputs[0]["label"],
            'score': outputs[0]["score"],
            'search_image': rec_search_image, # debug
            'template_image': self.rect_template_image # debug
               }

def build_tracker(model, postprocess, args):
    return Tracker(model, postprocess, args.exemplar_size, args.search_size, args.context_amount)
