
import argparse
import datetime
import json
import random
import time
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from dataset import build as build_dataset

sys.path.append('..')
import util.misc as utils
from util.box_ops import  box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

import cv2
import torchvision.transforms as T

def get_args_parser():
    parser = argparse.ArgumentParser('Test VID  dataset ', add_help=False)

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epoch', default=1, type=int)

    # dataset parameters
    parser.add_argument('--dataset_paths', default=[], nargs='+') # the path to datasets

    parser.add_argument('--dataset_video_frame_ranges', default=[100], nargs='+')
    parser.add_argument('--dataset_num_uses', default=[-1], nargs='+')
    parser.add_argument('--eval_dataset_num_uses', default=[], nargs='+')
    parser.add_argument('--template_aug_shift', default=4, type=int)
    parser.add_argument('--template_aug_scale', default=0.05, type=float)
    parser.add_argument('--template_aug_color', default=1.0, type=float)
    parser.add_argument('--search_aug_shift', default=64, type=int)
    parser.add_argument('--search_aug_scale', default=0.18, type=float)
    parser.add_argument('--search_aug_blur', default=0.2, type=float)
    parser.add_argument('--search_aug_color', default=1.0, type=float)
    parser.add_argument('--exempler_size', default=127, type=int)
    parser.add_argument('--search_size', default=255, type=int)
    parser.add_argument('--resnet_dilation', action='store_false',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)") #defualt is true
    parser.add_argument('--negative_aug_rate', default=0.2, type=float)
    parser.add_argument('--multi_template_num', default=1, type=int)

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

        for i, obj in enumerate(data_loader_train):
            print("{} iterator has {} batches".format(i, len(obj[-1])))
            targets = [{k: v.to(device) for k, v in t.items()} for t in obj[-1]]

            multi_frame_num = len(obj[0]) // len(obj[1])
            template_nested_samples = utils.nested_tensor_from_tensor_list(obj[0], obj[2])
            search_nested_samples = utils.nested_tensor_from_tensor_list(obj[1], obj[3])
            template_samples = template_nested_samples.tensors.to(device) # use several time to load to gpu
            search_samples = search_nested_samples.tensors.to(device)  # use several time to load to gpu
            template_masks = template_nested_samples.mask.to(device) # use several time to load to gpu
            search_masks = search_nested_samples.mask.to(device) # use several time to load to gpu


            # print(targets) # debug
            image_revert = T.Compose([
                T.Normalize(
                    -np.array(dataset_train.image_normalize.transforms[1].mean)
                    / np.array(dataset_train.image_normalize.transforms[1].std),
                    1 / np.array(dataset_train.image_normalize.transforms[1].std)),
                T.Normalize(0, [1.0 / 255, 1.0 / 255, 1.0 / 255])
            ])

            for batch_i in range(len(obj[-1])):

                revert_search_image = image_revert(search_samples[batch_i])
                search_image = revert_search_image.to('cpu').detach().numpy().copy()
                search_image = (np.round(search_image.transpose(1,2,0))).astype(np.uint8).copy()

                template_images = []
                for id in range(multi_frame_num):
                    revert_template_image = image_revert(template_samples[batch_i * multi_frame_num + id])
                    template_image = revert_template_image.to('cpu').detach().numpy().copy()
                    template_image = (np.round(template_image.transpose(1,2,0))).astype(np.uint8).copy()

                    mask = torch.round(obj[2][batch_i * multi_frame_num + id].to('cpu')).int()
                    cv2.rectangle(template_image, (mask[0], mask[1]), (mask[2], mask[3]), (0,255,0))
                    template_images.append(template_image)

                output_size = targets[batch_i]["hm"].shape[0]
                sesarch_size = search_image.shape[0]

                ind = targets[batch_i]["ind"].item()
                ct = torch.as_tensor([ind % output_size, ind // output_size], dtype=torch.float32)

                revert_ct = (ct + targets[batch_i]["reg"].to('cpu')) * float(sesarch_size) / float(output_size)
                revert_wh = targets[batch_i]["wh"].to('cpu') * float(sesarch_size)
                revert_bbox = torch.round(box_cxcywh_to_xyxy(torch.cat([revert_ct, revert_wh]))).int()
                # draw the reverted bbox from the target info
                cv2.rectangle(search_image, (revert_bbox[0], revert_bbox[1]), (revert_bbox[2], revert_bbox[3]), (0,0,255), 2)
                # draw the orginal ground truth bbox
                orig_bbox = torch.round(targets[batch_i]["bbox_debug"].to('cpu')).int()
                cv2.rectangle(search_image, (orig_bbox[0], orig_bbox[1]), (orig_bbox[2], orig_bbox[3]), (0,255,0))
                cv2.circle(search_image, (np.round(revert_ct[0]), np.round(revert_ct[1])), 2, (0,255,0), -1) # no need

                search_mask = torch.round(obj[3][batch_i].to('cpu')).int()
                cv2.rectangle(search_image, (search_mask[0], search_mask[1]), (search_mask[2], search_mask[3]), (0,255,0))

                #print("search_mask_float : {}, orig_bbox_float: {}".format(targets[batch_i]["search_mask"].to('cpu'), targets[batch_i]["bbox_debug"].to('cpu')))
                #print("search_mask: {}, orig_bbox: {}".format(search_mask, orig_bbox))

                heatmap = (torch.round(targets[batch_i]["hm"] * 255)).to('cpu').detach().numpy().astype(np.uint8)
                # print("center of gaussian peak: {}".format(np.unravel_index(np.argmax(heatmap), heatmap.shape)))
                # mask the heatmap to the original image
                heatmap_resize = cv2.resize(heatmap, search_image.shape[1::-1])
                # print("heatmap peal: {}, revert_wh: {}".format(np.unravel_index(np.argmax(heatmap_resize), heatmap_resize.shape), revert_ct))
                heatmap_color = np.stack([heatmap_resize, np.zeros(search_image.shape[1::-1], dtype=np.uint8), heatmap_resize], -1)
                search_image = np.round(0.4 * heatmap_color + 0.6 * search_image.copy()).astype(np.uint8)

                cv2.imshow('heatmap', heatmap)
                cv2.imshow('search_image', search_image)
                for i in range(multi_frame_num):
                    cv2.imshow('template' + str(i+1) + '_image', template_images[i])

                    template_image2 = deepcopy(template_images[i])
                    template_mask_img = template_masks[batch_i * multi_frame_num + id].to('cpu').detach().numpy()
                    template_image2[np.repeat(template_mask_img[:, :, np.newaxis], 3, axis=2)] = 0
                    cv2.imshow('template' + str(i + 1) + '_image_mask', template_image2)

                search_image2 = deepcopy(search_image)
                search_mask_img = search_masks[batch_i].to('cpu').detach().numpy()
                search_image2[np.repeat(search_mask_img[:, :, np.newaxis], 3, axis=2)] = 0
                cv2.imshow('search_image_mask', search_image2)

                k = cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
