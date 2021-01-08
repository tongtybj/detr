# Copyright (c) SenseTime. All Rights Reserved.

'''
usage:

python test.py --dataset_paths ./yt_bb/youtube_bb/Curation ./vid/ILSVRC2015/Curation/ --dataset_video_frame_ranges 100 3 --dataset_num_uses 20 20

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from importlib import import_module
import mimetypes
import logging
import sys
import os
from os.path import join

import pathlib
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as T
import math

try:
    from .augmentation import Augmentation, corner2center, center2corner, Center
    from .utils import gaussian_radius, draw_umich_gaussian
except ImportError as e:
    if "attempted relative import with no known parent package" == str(e):
         # for test.py
        from augmentation import Augmentation, corner2center, center2corner, Center
        from utils import gaussian_radius, draw_umich_gaussian

logger = logging.getLogger("global")


class SubDataset(object):
    def __init__(self, path, ann_file, frame_range, num_use, start_idx, multi_template_num):

        self.path = path
        self.ann_file = ann_file
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        self.multi_template_num = multi_template_num
        logger.info("loading " + self.path)


        ext = str(self.ann_file).split(".")[-1]
        read_type = 'rb' if ext == "pickle" else 'r'
        try:
            module = import_module(ext)

            with open(self.ann_file, read_type) as f:

                meta_data = module.load(f)
                meta_data = self._filter_zero(meta_data)
        except:
            raise TypeError("can not load module {}, currently can only support json and pickle".format(ext))

        for video in list(meta_data.keys()):
            for track in meta_data[video]["tracks"]:
                frames = meta_data[video]["tracks"][track]
                # print("frames for {}: {}".format(video, frames.keys()))
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video]["tracks"][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video]["tracks"][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]["tracks"]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        #print("size of meta_data for {} is {}".format(ann_file, len(meta_data)))
        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.path))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            new_tracks['frame_size'] = tracks['frame_size']
            new_tracks['tracks'] = {}
            for trk, frames in tracks['tracks'].items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks['tracks'][trk] = new_frames
            if len(new_tracks['tracks']) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.path, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))

        pick = []
        # assume self.num_use > self.num
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]


    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.path, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video]["tracks"][track][frame]
        image_size = self.labels[video]["frame_size"]
        return image_path, image_anno, image_size

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video["tracks"].keys())) # random choose a track id from a video
        track_info = video["tracks"][track]

        frames = track_info['frames']
        template_frame_id = np.random.randint(0, len(frames))
        template_frame = frames[template_frame_id]

        left = max(template_frame_id - self.frame_range, 0)
        right = min(template_frame_id + self.frame_range, len(frames)-1) + 1
        search_frame_id = np.random.randint(left, right)
        search_frame = frames[search_frame_id]
        search_frame_info = self.get_image_anno(video_name, track, search_frame)

        #print("template_frame: {}".format(template_frame))
        template_frame_info_list = [self.get_image_anno(video_name, track, template_frame)]
        end_frame_id = search_frame_id
        for i in range(self.multi_template_num - 1):
            if end_frame_id < template_frame_id:
                intermediate_frame = np.random.choice(frames[end_frame_id + 1:template_frame_id + 1])
            elif search_frame_id > template_frame_id:
                intermediate_frame = np.random.choice(frames[template_frame_id:end_frame_id])
            else:
                intermediate_frame = template_frame

            template_frame_info_list.append(self.get_image_anno(video_name, track, intermediate_frame))
            end_frame_id = intermediate_frame
            #print("intermediate_frame: {}".format(intermediate_frame))

        #print("search_frame: {}".format(search_frame))

        return template_frame_info_list, search_frame_info

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video["tracks"].keys()))
        track_info = video["tracks"][track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self,
                 image_set, dataset_paths,
                 dataset_video_frame_ranges, dataset_num_uses,
                 template_shift, template_scale, template_color,
                 search_shift, search_scale, search_blur, search_color,
                 exempler_size = 127, search_size = 255,
                 negative_rate = 0.5,
                 resnet_dilation = [],
                 multi_template_num = 1):
        super(TrkDataset, self).__init__()

        self.exempler_size = exempler_size
        self.search_size = search_size
        self.negative_rate = negative_rate

        # woraround for resnet:
        if resnet_dilation:
            resnet_dilation = [False, True, True]
        else:
            resnet_dilation = [False, False, False]

        stride = 4
        for dilation in resnet_dilation:
            if not dilation:
                stride = stride * 2
        self.output_size = (self.search_size + stride - 1) // stride

        self.multi_template_num = multi_template_num # how many template frames for single search frame

        # debug (TODO remove)
        if self.search_size == 255:
            assert self.output_size == 32

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0

        for path, video_frame_range, num_use in zip(dataset_paths, dataset_video_frame_ranges, dataset_num_uses):

            assert Path(path).exists(), f'provided VID path {path} does not exist'

            ANN_PATHS = {
                "train": join(path, 'train.pickle'),
                "val":   join(path, 'val.pickle')
            }

            sub_dataset = SubDataset(
                path, ANN_PATHS[image_set],
                int(video_frame_range),
                int(num_use),
                start,
                multi_template_num)
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)


        print("size of {} dataset: {}".format(image_set, self.num))
        self.pick = self.shuffle()

        # data augmentation
        self.template_aug = Augmentation(
            template_shift,
            template_scale,
            0,
            0,
            template_color)
        self.search_aug = Augmentation(
            search_shift,
            search_scale,
            search_blur,
            0,
            search_color)

        # input image scaling for the backbone pretrained with ImageNet
        # https://pytorch.org/docs/stable/torchvision/models.html
        self.image_normalize = T.Compose([
            T.ToTensor(), # Converts a numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def normalize(self, image, bbox = None):
        image = self.image_normalize(image)

        if not bbox:
            return image
        else:
            h, w = image.shape[-2:]
            bbox = torch.as_tensor(corner2center(bbox), dtype=torch.float32)
            bbox = bbox / torch.tensor([w, h, w, h], dtype=torch.float32)
            return image, bbox

    def shuffle(self):
        pick = []
        for sub_dataset in self.all_dataset:
            pick += sub_dataset.pick

        assert len(pick) == self.num
        logger.info("dataset length {}".format(self.num))
        return pick

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape, raw_img_size):
        imh, imw = image.shape[:2]
        assert imh == imw
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.exempler_size
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        s_x = imh / scale_z
        bb_center = [(shape[2]+shape[0])/2., (shape[3]+shape[1])/2.]
        mask = [0, 0, imw, imh]
        if bb_center[0] < s_x/2:
            mask[0] = (s_x/2 - bb_center[0]) * scale_z
        if bb_center[1] < s_x/2:
            mask[1] = (s_x/2 - bb_center[1]) * scale_z
        if bb_center[0] + s_x/2 > raw_img_size[0]:
            mask[2] = mask[2] - (bb_center[0] + s_x/2 - raw_img_size[0]) * scale_z
        if bb_center[1] + s_x/2 > raw_img_size[1]:
            mask[3] = mask[3] - (bb_center[1] + s_x/2 - raw_img_size[1]) * scale_z


        w = w*scale_z
        h = h*scale_z
        cx, cy = imw/2., imh/2. # TODO: original is imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))

        #print("raw shape: {}, raw bbox : {}, raw mask: {}".format(shape, bbox, mask))

        return bbox, mask

    def __len__(self):
        return self.num

    def __getitem__(self, index):

        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        # gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()

        # negative sample
        neg = self.negative_rate and self.negative_rate > np.random.random()

        if neg:
            templates_info = [dataset.get_random_target(index) for i in range(self.multi_template_num)]
            search_info = np.random.choice(self.all_dataset).get_random_target()
        else:
            templates_info, search_info = dataset.get_positive_pair(index)

        # template augmentation
        templates = []
        templates_mask = []
        for i in range(self.multi_template_num):
            image = cv2.imread(templates_info[i][0]) # get image
            box, mask  = self._get_bbox(image, templates_info[i][1], templates_info[i][2])
            template, _ , mask = self.template_aug(image, box, mask, self.exempler_size)

            # normalize
            # we have to change to type from float to uint8 for torchvision.transforms.ToTensor
            # https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor
            template = self.normalize(np.round(template).astype(np.uint8))
            mask = torch.as_tensor(mask)

            templates.append(template)
            templates_mask.append(mask)

        search_image = cv2.imread(search_info[0])
        search_box, search_mask = self._get_bbox(search_image, search_info[1], search_info[2])
        search, input_bbox, search_mask = self.search_aug(search_image,
                                                          search_box,
                                                          search_mask,
                                                          self.search_size)
        # normalize
        # we have to change to type from float to uint8 for torchvision.transforms.ToTensor
        # https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor
        search = self.normalize(np.round(search).astype(np.uint8))
        search_mask =  torch.as_tensor(search_mask)


        hm = np.zeros((self.output_size, self.output_size), dtype=np.float32)

        valid = not neg
        scale = float(self.output_size) / float(self.search_size)
        output_bbox = [input_bbox.x1 * scale, input_bbox.y1 * scale, input_bbox.x2 * scale, input_bbox.y2 * scale]
        radius = gaussian_radius((math.ceil(output_bbox[3] - output_bbox[1]), math.ceil(output_bbox[2] - output_bbox[0])))
        radius = max(0, int(radius))
        ct = np.array([(output_bbox[0] + output_bbox[2]) / 2, (output_bbox[1] + output_bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        reg = torch.tensor(ct - ct_int, dtype=torch.float32) # range is [0, 1)
        wh = torch.tensor([input_bbox[2] - input_bbox[0], input_bbox[3] - input_bbox[1]], dtype=torch.float32) / float(self.search_size) # normalized
        ind = ct_int[1] * self.output_size + ct_int[0]
        if valid:
            draw_umich_gaussian(hm, ct_int, radius)
        hm = torch.tensor(hm, dtype=torch.float32) # range is [0, 1)


        # print("index: {}, gaussian radius: {}, ct: {}, reg: {}, wh: {}, valid: {}".format(index, radius, ct, reg, wh, valid)) # debug

        """
        print("dataset idx: {}".format(index))
        print("aug search image for {}: {}".format(index, search))
        temp_search = search.astype(np.uint8).copy()
        bbox_int = np.round(bbox).astype(np.uint16)
        cv2.rectangle(temp_search, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), (0,255,0))

        cv2.imshow('auged search_image', temp_search)
        k = cv2.waitKey(0)
        """

        target_dict =  {'hm': hm, 'reg': reg, 'wh': wh, 'ind': torch.as_tensor(ind), 'valid': torch.as_tensor(valid), 'bbox_debug': torch.as_tensor(input_bbox)}

        return templates, search, templates_mask, search_mask, target_dict

def build(image_set, args):

    assert len(args.dataset_paths) == len(args.dataset_video_frame_ranges) == len(args.dataset_num_uses)

    if image_set == 'val':
        if len(args.dataset_num_uses) == len(args.eval_dataset_num_uses):
            args.dataset_num_uses = args.eval_dataset_num_uses
        else:
            args.dataset_num_uses = [-1] * len(args.dataset_num_uses)

    dataset = TrkDataset(image_set, args.dataset_paths,
                         args.dataset_video_frame_ranges, args.dataset_num_uses,
                         args.template_aug_shift, args.template_aug_scale, args.template_aug_color,
                         args.search_aug_shift, args.search_aug_scale, args.search_aug_blur, args.search_aug_color,
                         args.exempler_size, args.search_size,
                         args.negative_aug_rate,
                         args.resnet_dilation,
                         args.multi_template_num)
    return dataset
