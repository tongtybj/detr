
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

# hack
import sys
sys.path.append('../../')
from datasets import build_dataset
import util.misc as utils
from util.box_ops import  box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

import cv2
import torchvision.transforms as T

def get_args_parser():
    parser = argparse.ArgumentParser('Test VID  dataset ', add_help=False)

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epoch', default=1, type=int)

    # dataset parameters
    parser.add_argument('--dataset_file', default='vid')
    parser.add_argument('--vid_path', type=str)

    # vid dataset parameters
    parser.add_argument('--video_frame_range', default=100, type=int)
    parser.add_argument('--template_aug_shift', default=4, type=int)
    parser.add_argument('--template_aug_scale', default=0.05, type=float)
    parser.add_argument('--template_aug_color', default=1.0, type=float)
    parser.add_argument('--search_aug_shift', default=64, type=int)
    parser.add_argument('--search_aug_scale', default=0.18, type=float)
    parser.add_argument('--search_aug_color', default=1.0, type=float)
    parser.add_argument('--exempler_size', default=127, type=int)
    parser.add_argument('--search_size', default=255, type=int)
    parser.add_argument('--negative_aug_rate', default=0.8, type=float)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):

    if not args.vid_path:
        raise NameError("Please assign the path to get VID raw data using --vid_path")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    for epoch in range(args.epoch):

        print("the len of dataset_train: {}".format(len(dataset_train)))

        for i,obj in enumerate(data_loader_train):
            print("{} iterator has {} batches".format(i, len(obj[2])))
            template_samples = obj[0].to(device) # use several time to load to gpu
            search_samples = obj[1].to(device)  # use several time to load to gpu
            targets = [{k: v.to(device) for k, v in t.items()} for t in obj[2]]  # use several time to load to gpu


            # debug
            print(targets)
            image_revert = T.Compose([
                T.Normalize(
                    -np.array(dataset_train.image_normalize.transforms[1].mean)
                    / np.array(dataset_train.image_normalize.transforms[1].std),
                    1 / np.array(dataset_train.image_normalize.transforms[1].std)),
                T.Normalize(0, [1.0 / 255, 1.0 / 255, 1.0 / 255])
            ])

            for batch_i in range(len(obj[2])):
                revert_search_image = image_revert(search_samples.tensors[batch_i])
                search_image = revert_search_image.to('cpu').detach().numpy().copy()
                search_image = (np.round(search_image.transpose(1,2,0))).astype(np.uint8).copy()

                revert_template_image = image_revert(template_samples.tensors[batch_i])
                template_image = revert_template_image.to('cpu').detach().numpy().copy()
                template_image = (np.round(template_image.transpose(1,2,0))).astype(np.uint8).copy()

                print("bbox: {}".format(targets[batch_i]["bbox"]))
                h, w = search_image.shape[:2]
                xyxy = box_cxcywh_to_xyxy(targets[batch_i]["bbox"]).to("cpu")
                bbox_int = (torch.round(xyxy * torch.tensor([w, h, w, h]))).int()
                print(type(search_image))
                cv2.rectangle(search_image, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), (0,255,0))

                cv2.imshow('search_image', search_image)
                cv2.imshow('template_image', template_image)
                k = cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
