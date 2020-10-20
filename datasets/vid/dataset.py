# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from importlib import import_module
import mimetypes
import logging
import sys
import os

import pathlib
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as T
from .augmentation import Augmentation, corner2center, center2corner, Center

logger = logging.getLogger("global")


class SubDataset(object):
    def __init__(self, name, root, ann_file, frame_range, num_use, start_idx):
        self.name = name
        self.root = root
        self.ann_file = ann_file
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)


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
            for track in meta_data[video]:
                frames = meta_data[video][track]
                # print("frames for {}: {}".format(video, frames.keys()))
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        #print("size of meta_data for {} is {}".format(ann_file, len(meta_data)))
        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'


    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
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
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))


    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self, name, root, ann_file, video_frame_range,
                 template_shift, template_scale, template_color,
                 search_shift, search_scale, search_color,
                 exempler_size = 127, search_size = 255,
                 negative_rate = 0.5):
        super(TrkDataset, self).__init__()

        self.exempler_size = exempler_size
        self.search_size = search_size
        self.negative_rate = negative_rate

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0

        # TODO, re-implement for multiple dataset
        sub_dataset = SubDataset(
            name,
            root,
            ann_file,
            video_frame_range,
            -1,
            start)
        start += sub_dataset.num
        self.num += sub_dataset.num_use

        sub_dataset.log()
        self.all_dataset.append(sub_dataset)

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
            0,
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

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
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
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):

        # only do for odd index
        neg = 0
        if index % 2 == 1:
            neg = self.negative_rate and self.negative_rate > np.random.random()

        dataset, index = self._find_dataset(index)

        # gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()

        # get one dataset
        if neg:
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        # debug
        #if index == 0:
        #    cv2.imshow('raw search_image', search_image)

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        self.exempler_size)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       self.search_size)



        """
        print("dataset idx: {}".format(index))
        print("aug search image for {}: {}".format(index, search))
        if index == 0:
            temp_search = search.astype(np.uint8).copy()
            bbox_int = np.round(bbox).astype(np.uint16)
            cv2.rectangle(temp_search, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), (0,255,0))

            cv2.imshow('auged search_image', temp_search)
            k = cv2.waitKey(0)
        """

        # normalize
        # we have to change to type from float to uint8 for torchvision.transforms.ToTensor
        # https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor
        template = self.normalize(np.round(template).astype(np.uint8))
        search, bbox = self.normalize(np.round(search).astype(np.uint8), bbox)

        # print("aug search image for {}: {}".format(index, search))

        target_dict =  {
            'label': torch.as_tensor([1 if neg == 0 else 0]), # 0: non-object, 1: object
            'bbox': bbox,
            'index': torch.as_tensor([index])
        }

        return template, search, target_dict

def build(image_set, args):
    root = Path(args.vid_path)
    assert root.exists(), f'provided VID path {root} does not exist'

    """
    ANN_PATHS = {
        "train": root / f'train.json',
        "val": root / f'val.json'
    }
    """
    ANN_PATHS = {
        "train": root / f'train.pickle',
        "val": root / f'val.pickle'
    }

    dataset = TrkDataset("vid", root, ANN_PATHS[image_set], args.video_frame_range,
                         args.template_aug_shift, args.template_aug_scale, args.template_aug_color,
                         args.search_aug_shift, args.search_aug_scale, args.search_aug_color,
                         args.exempler_size, args.search_size,
                         args.negative_aug_rate)
    return dataset
