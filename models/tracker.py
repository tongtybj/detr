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
from datasets.utils import crop_image, siamfc_like_scale, get_exemplar_size, get_context_amount # TODO: move to utils
from util.box_ops import box_cxcywh_to_xyxy
from util.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
import cv2
import time

class Tracker(object):
    def __init__(self, model, postprocess, search_size, window_factor, score_threshold, window_steps, size_penalty_k, size_lpf):
        self.model = model
        self.model.eval()
        self.postprocess = postprocess

        self.search_size = search_size
        backbone_stride = model.backbone.stride
        self.heatmap_size = (search_size + backbone_stride - 1) // backbone_stride
        self.size_lpf = size_lpf
        self.size_penalty_k = size_penalty_k

        hanning = np.hanning(self.heatmap_size)
        self.window = torch.as_tensor(np.outer(hanning, hanning).flatten())
        self.window_factor = window_factor
        self.score_threshold = score_threshold
        self.window_steps = window_steps

        self.init_bbox_size = []

        self.image_normalize = T.Compose([
            T.ToTensor(), # Converts a numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.template_update = False

        self.rect_template_image = None
        self.prev_img = None
        self.prev_img_update = False
        self.prev_rect_template_image = None

        self.num_segment = self.model.segment_embedding.num_segment
        if self.num_segment > 2:
            raise ValueError("only can support multi frame template model with single or two frames")

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
        self.center_pos = torch.Tensor([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2])
        self.size = torch.Tensor(bbox[2:])
        channel_avg = np.mean(img, axis=(0, 1))

        # get crop
        s_z, scale_z = siamfc_like_scale(bbox_xyxy)
        template_image, _ = crop_image(img, bbox_xyxy, padding = channel_avg)

        self.rect_template_image = template_image.copy()
        init_bbox = np.array(self.size) * scale_z
        exemplar_size = get_exemplar_size()
        x1 = np.round(exemplar_size/2 - init_bbox[0]/2).astype(np.uint8)
        y1 = np.round(exemplar_size/2 - init_bbox[1]/2).astype(np.uint8)
        x2 = np.round(exemplar_size/2 + init_bbox[0]/2).astype(np.uint8)
        y2 = np.round(exemplar_size/2 + init_bbox[1]/2).astype(np.uint8)
        cv2.rectangle(self.rect_template_image, (x1, y1), (x2, y2), (0,255,0), 3)

        # get mask
        self.init_template_mask = [0, 0, template_image.shape[0], template_image.shape[1]]
        if self.center_pos[0] < s_z/2:
            self.init_template_mask[0] = (s_z/2 - self.center_pos[0]) * scale_z
        if self.center_pos[1] < s_z/2:
            self.init_template_mask[1] = (s_z/2 - self.center_pos[1]) * scale_z
        if self.center_pos[0] + s_z/2 > img.shape[1]:
            self.init_template_mask[2] = self.init_template_mask[2] - (self.center_pos[0] + s_z/2 - img.shape[1]) * scale_z
        if self.center_pos[1] + s_z/2 > img.shape[0]:
            self.init_template_mask[3] = self.init_template_mask[3] - (self.center_pos[1] + s_z/2 - img.shape[0]) * scale_z


        # normalize and conver to torch.tensor
        self.init_template = self.image_normalize(np.round(template_image).astype(np.uint8)).cuda()
        self.template_update = True

        self.init_best_score = 0

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
        channel_avg = np.mean(img, axis=(0, 1)) # TODO: no need?
        _, search_image = crop_image(img, bbox_xyxy, padding = channel_avg, instance_size = self.search_size)
        s_z, scale_z = siamfc_like_scale(bbox_xyxy)
        s_x = self.search_size / scale_z
        # get mask
        search_mask = [0, 0, search_image.shape[0], search_image.shape[1]]
        if self.center_pos[0] < s_x/2:
            search_mask[0] = (s_x/2 - self.center_pos[0]) * scale_z
        if self.center_pos[1] < s_x/2:
            search_mask[1] = (s_x/2 - self.center_pos[1]) * scale_z
        if self.center_pos[0] + s_x/2 > img.shape[1]:
            search_mask[2] = search_mask[2] - (self.center_pos[0] + s_x/2 - img.shape[1]) * scale_z
        if self.center_pos[1] + s_x/2 > img.shape[0]:
            search_mask[3] = search_mask[3] - (self.center_pos[1] + s_x/2 - img.shape[0]) * scale_z

        # normalize and conver to torch.tensor
        search = self.image_normalize(np.round(search_image).astype(np.uint8)).cuda()

        template_list = [self.init_template] * self.num_segment
        template_mask_list = [torch.as_tensor(self.init_template_mask).float()] * self.num_segment

        if self.num_segment == 2 and self.prev_img_update:

            self.prev_img_update = False

            channel_avg = np.mean(self.prev_img, axis=(0, 1))
            prev_template_image, _ = crop_image(self.prev_img, bbox_xyxy, padding = channel_avg)

            # normalize and conver to torch.tensor
            prev_template = self.image_normalize(np.round(prev_template_image).astype(np.uint8)).cuda()

            # get mask
            s_z, scale_z = siamfc_like_scale(bbox_xyxy)
            prev_template_mask = [0, 0, prev_template_image.shape[0], prev_template_image.shape[1]]
            if self.center_pos[0] < s_z/2:
                prev_template_mask[0] = (s_z/2 - self.center_pos[0]) * scale_z
            if self.center_pos[1] < s_z/2:
                prev_template_mask[1] = (s_z/2 - self.center_pos[1]) * scale_z
            if self.center_pos[0] + s_z/2 > img.shape[1]:
                prev_template_mask[2] = prev_template_mask[2] - (self.center_pos[0] + s_z/2 - img.shape[1]) * scale_z
            if self.center_pos[1] + s_z/2 > img.shape[0]:
                prev_template_mask[3] = prev_template_mask[3] - (self.center_pos[1] + s_z/2 - img.shape[0]) * scale_z

            template_list[-1] = prev_template
            template_mask_list[-1] = torch.as_tensor(prev_template_mask).float()


            # visualize debug
            self.prev_rect_template_image = prev_template_image.copy()
            init_bbox = np.array(self.size) * scale_z
            exemplar_size = get_exemplar_size()
            x1 = np.round(exemplar_size/2 - init_bbox[0]/2).astype(np.uint8)
            y1 = np.round(exemplar_size/2 - init_bbox[1]/2).astype(np.uint8)
            x2 = np.round(exemplar_size/2 + init_bbox[0]/2).astype(np.uint8)
            y2 = np.round(exemplar_size/2 + init_bbox[1]/2).astype(np.uint8)
            cv2.rectangle(self.prev_rect_template_image, (x1, y1), (x2, y2), (0,255,0), 3)

            self.template_update = True
            #print("update template images")

        with torch.no_grad():
            if self.template_update:
                outputs = self.model(nested_tensor_from_tensor_list([search], [torch.as_tensor(search_mask).float()]),
                                     nested_tensor_from_tensor_list(template_list, template_mask_list))
                self.template_update = False
            else:
                outputs = self.model(nested_tensor_from_tensor_list([search], [torch.as_tensor(search_mask).float()]))


        outputs = self.postprocess(outputs)

        heatmap = outputs['pred_heatmap'][0].cpu() # we only address with a single image


        assert heatmap.size(0) == self.heatmap_size * self.heatmap_size
        raw_heatmap = heatmap.view(self.heatmap_size, self.heatmap_size) # as a image format
        # print("postprocess raw heatmap shape: {}".format(raw_heatmap.shape))

        if torch.max(heatmap) > self.score_threshold:


            def change(r):
                return torch.max(r, 1. / r)

            bbox_wh_map = outputs['pred_bbox_wh'][0].cpu() * torch.as_tensor(search.shape[-2:])  # convert from relative [0, 1] to absolute [0, height] coordinates

            # scale penalty
            pad = (bbox_wh_map[:, 0] + bbox_wh_map[:, 1]) * get_context_amount()
            sz = torch.sqrt((bbox_wh_map[:, 0] + pad) * (bbox_wh_map[:, 1] + pad))
            s_c = change(sz / get_exemplar_size())

            # aspect ratio penalty
            r_c = change((bbox_wh_map[:, 0] / bbox_wh_map[:, 1]) / (self.size[0] / self.size[1]) )
            penalty = torch.exp(-(r_c * s_c - 1) * self.size_penalty_k)

            best_idx = 0
            window_factor = self.window_factor
            post_heatmap = None
            best_score = 0
            for i in range(self.window_steps):
                # add distance penalty
                post_heatmap = penalty * heatmap * (1 -  window_factor) + self.window * window_factor
                best_idx = torch.argmax(post_heatmap)
                best_score = heatmap[best_idx].item()

                if best_score > self.score_threshold:
                    break;
                else:
                    window_factor = np.max(window_factor - self.window_factor / self.window_steps, 0)
                    #print("reduce the window factor from {} to {}".format(window_factor + self.window_factor / self.window_steps, window_factor))

            post_heatmap = post_heatmap.view(self.heatmap_size, self.heatmap_size) # as a image format

            # print("postprocess best score: {}".format(best_score))
            # print("postprocess post heatmap shape: {}".format(post_heatmap.shape))

            # bbox
            ct_int = torch.stack([best_idx % self.heatmap_size, best_idx // self.heatmap_size], dim = -1)
            bbox_reg = outputs['pred_bbox_reg'][0][best_idx].cpu()
            bbox_ct = (ct_int + bbox_reg) * torch.as_tensor(search.shape[-2:]) / float(self.heatmap_size)
            bbox_wh = bbox_wh_map[best_idx]
            # print("postprocess best idx {}, ct_int: {}, reg: {}, ct: {}, and wh: {}".format(best_idx, ct_int, bbox_reg, bbox_ct, bbox_wh))


            ct_delta = (bbox_ct - self.search_size / 2) / scale_z
            cx = self.center_pos[0] + ct_delta[0].item()
            cy = self.center_pos[1] + ct_delta[1].item()

            # smooth bbox
            lpf = best_score * self.size_lpf
            bbox_wh = bbox_wh / scale_z
            width = self.size[0] * (1 - lpf) + bbox_wh[0].item() * lpf
            height = self.size[1] * (1 - lpf) + bbox_wh[1].item() * lpf

            # clip boundary
            bbox = [cx - width / 2, cy - height / 2,
                    cx + width / 2, cy + height / 2]
            bbox = self._bbox_clip(bbox, img.shape[:2])

            # udpate state
            self.center_pos = torch.Tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            self.size = torch.Tensor([bbox[2] - bbox[0], bbox[3] - bbox[1]])

            # debug for search image:
            start_time = time.time()
            debug_bbox = box_cxcywh_to_xyxy(torch.round(torch.cat([bbox_ct, bbox_wh])).int())
            rec_search_image = cv2.rectangle(search_image,
                                             (debug_bbox[0], debug_bbox[1]),
                                             (debug_bbox[2], debug_bbox[3]),(0,255,0),3)

            raw_heatmap = (torch.round(raw_heatmap * 255)).detach().numpy().astype(np.uint8)
            post_heatmap = (torch.round(post_heatmap * 255)).detach().numpy().astype(np.uint8)
            heatmap_resize = cv2.resize(raw_heatmap, search_image.shape[1::-1])
            heatmap_color = np.stack([heatmap_resize, np.zeros(search_image.shape[1::-1], dtype=np.uint8), heatmap_resize], -1)
            rec_search_image = np.round(0.4 * heatmap_color + 0.6 * rec_search_image.copy()).astype(np.uint8)
            # print("postprocess time: {}".format(time.time() - start_time))

            if self.init_best_score == 0:
                self.init_best_score = best_score

            if best_score > self.init_best_score:
                self.prev_img_update = True
                self.prev_img = img.copy()
        else:
            bbox = [self.center_pos[0] - self.size[0] / 2, self.center_pos[1] - self.size[1] / 2,
                    self.center_pos[0] + self.size[0] / 2, self.center_pos[1] + self.size[1] / 2]
            best_score = self.score_threshold
            raw_heatmap = (torch.round(raw_heatmap * 255)).detach().numpy().astype(np.uint8)
            post_heatmap = raw_heatmap
            rec_search_image = search_image



        return {
            'bbox': bbox,
            'score': best_score,
            'raw_heatmap': raw_heatmap,
            'post_heatmap': post_heatmap,
            'search_image': rec_search_image, # debug
            'template_image': self.rect_template_image, # debug
            'prev_template_image': self.prev_rect_template_image # debug
        }

def build_tracker(model, postprocess, args):
    return Tracker(model, postprocess, args.search_size, args.window_factor, args.score_threshold, args.window_steps, args.tracking_size_penalty_k, args.tracking_size_lpf)
