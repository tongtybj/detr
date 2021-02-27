# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import sys
import cv2
import copy
import torch
import numpy as np
from glob import glob

sys.path.append('..')
from util.box_ops import get_axis_aligned_bbox
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

from models.tracker import build_tracker as build_baseline_tracker
from models.hybrid_tracker import build_tracker as build_online_tracker

def get_args_parser():
    parser = argparse.ArgumentParser('benchmark dataset inference', add_help=False)

    # Model parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # * Backbone
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--resnet_dilation', action='store_false',
                        help="If true (default), we replace stride with dilation in resnet blocks") #default is true
    parser.add_argument('--return_interm_layers', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer") # switch by eval() / train()
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--return_layers', default=[], nargs='+')
    parser.add_argument('--dcf_layers', default=[], nargs='+')
    parser.add_argument('--weighted', action='store_true',
                        help="the weighted for the multiple input embedding for transformer")
    parser.add_argument('--transformer_mask', action='store_false',
                        help="mask for transformer") # workaround to enable transformer mask for default
    parser.add_argument('--multi_frame', action='store_true',
                        help="use multi frame for encoder (template images)")
    parser.add_argument('--repetition', default=1, type=int)
    parser.add_argument('--min_lost_rate_for_repeat', default=0.1, type=float) # change for different benchmark

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--loss_mask', action='store_true',
                        help="mask for heamtmap loss")


    # * Loss coefficients
    parser.add_argument('--reg_loss_coef', default=1, type=float,
                        help="weight (coeffficient) about bbox offset reggresion loss")
    parser.add_argument('--wh_loss_coef', default=1, type=float,
                        help="weight (coeffficient) about bbox width/height loss")

    # tracking
    parser.add_argument('--checkpoint', default="", type=str)
    parser.add_argument('--exemplar_size', default=127, type=int)
    parser.add_argument('--search_size', default=255, type=int)
    parser.add_argument('--context_amount', default=0.5, type=float)
    parser.add_argument('--use_baseline_tracker', action='store_true')

    # * hyper-parameter for tracking
    parser.add_argument('--score_threshold', default=0.05, type=float,
                        help='the lower score threshold to recognize a target (score_target > threshold) ')
    parser.add_argument('--window_steps', default=3, type=int,
                        help='the pyramid factor to gradually reduce the widow effect')
    parser.add_argument('--window_factor', default=0.4, type=float,
                        help='the factor of the hanning window for heatmap post process')
    parser.add_argument('--tracking_size_penalty_k', default=0.04, type=float,
                        help='the factor to penalize the change of size')
    parser.add_argument('--tracking_size_lpf', default=0.8, type=float,
                        help='the factor of the lpf for size tracking')
    parser.add_argument('--dcf_rate', default=0.8, type=float,
                        help='the weight for integrate dcf and trtr for heatmap ')
    parser.add_argument('--dcf_sample_memory_size', default=250, type=int,
                        help='the size of the trainining sample for DCF ')

    parser.add_argument('--dcf_size', default=0, type=int,
                        help='the size for feature for dcf')
    parser.add_argument('--boundary_recovery', action='store_true',
                        help='whether use boundary recovery')
    parser.add_argument('--hard_negative_recovery', action='store_true',
                        help='whether use hard negative recovery')
    parser.add_argument('--lost_target_recovery', action='store_true',
                        help='whether use lost target recovery')
    parser.add_argument('--lost_target_margin', default=0.3, type=float)
    parser.add_argument('--translation_threshold', default=0.03, type=float)
    parser.add_argument('--lost_target_cnt_threshold', default=60, type=int)
    parser.add_argument('--lost_target_score_threshold', default=0.5, type=float)


    parser.add_argument('--dataset_path', default="", type=str, help='path of datasets')
    parser.add_argument('--dataset', type=str, help='the benchmark', default="VOT2018")
    parser.add_argument('--video', default='', type=str, help='eval one special video')
    parser.add_argument('--vis', action='store_true', help='whether visualzie result')
    parser.add_argument('--debug_vis', action='store_true', help='wheterh visualize the debug result')
    parser.add_argument('--model_name', default='trtr', type=str)

    parser.add_argument('--result_path', default='results', type=str)
    parser.add_argument('--external_tracker', default="", type=str)


    return parser


def main(args, tracker):

    # create dataset
    if not args.dataset_path:
        args.dataset_path = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(args.dataset_path, 'dataset', args.dataset)

    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False,
                                            single_video=args.video)

    model_name = args.model_name

    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019', 'VOT2020']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue

            video_total_lost = 0
            for cnt in range(args.repetition):
                frame_counter = 0
                lost_number = 0
                toc = 0
                pred_bboxes = []

                template_image = None
                search_image = None
                raw_heatmap = None
                post_heatmap = None
                prev_search_image = None

                for idx, (img, gt_bbox) in enumerate(video):
                    if len(gt_bbox) == 4:
                        gt_bbox = [gt_bbox[0], gt_bbox[1],
                                   gt_bbox[0], gt_bbox[1]+gt_bbox[3],
                                   gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3],
                                   gt_bbox[0]+gt_bbox[2], gt_bbox[1]]
                    tic = cv2.getTickCount()
                    if idx == frame_counter:
                        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                        gt_bbox_ = [cx - w/2, cy - h/2, w, h]
                        tracker.init(img, gt_bbox_)
                        pred_bbox = gt_bbox_
                        pred_bboxes.append(1)
                    elif idx > frame_counter:
                        outputs = tracker.track(img)
                        pred_bbox = outputs['bbox']
                        pred_bbox = [pred_bbox[0], pred_bbox[1],
                                     pred_bbox[2] - pred_bbox[0],
                                     pred_bbox[3] - pred_bbox[1]]

                        overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                        if overlap > 0:
                            # not lost
                            pred_bboxes.append(pred_bbox)
                        else:
                            # lost object
                            pred_bboxes.append(2)
                            frame_counter = idx + 5 # skip 5 frames
                            lost_number += 1

                            if args.vis and args.debug_vis:


                                cv2.imshow('prev_search_image', prev_search_image)

                                cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 2))], True, (0, 255, 0), 3)

                                bbox = list(map(int, pred_bbox))
                                cv2.rectangle(img, (bbox[0], bbox[1]),
                                              (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                cv2.putText(img, 'lost', (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.imshow(video.name, img)

                                for key, value in outputs.items():
                                    if isinstance(value, np.ndarray):
                                        if len(value.shape) == 3 or len(value.shape) == 2:
                                            cv2.imshow(key, value)

                                k = cv2.waitKey(0)
                                if k == 27:         # wait for ESC key to exit
                                    sys.exit()

                    else:
                        pred_bboxes.append(0)
                    toc += cv2.getTickCount() - tic
                    if idx == 0:
                        if args.vis:
                            cv2.destroyAllWindows()
                    if args.vis and idx > frame_counter:
                        cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 2))], True, (0, 255, 0), 3)

                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                        cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow(video.name, img)

                        if args.debug_vis:

                            if prev_search_image is not None:
                                cv2.imshow('prev_search_image', prev_search_image)

                            for key, value in outputs.items():
                                if isinstance(value, np.ndarray):
                                    if len(value.shape) == 3 or len(value.shape) == 2:
                                        cv2.imshow(key, value)

                                        if key == 'search_image':
                                            prev_search_image = value

                            k = cv2.waitKey(0)
                            if k == 27:         # wait for ESC key to exit
                                break
                        else:
                            k = cv2.waitKey(1)
                            if k == 27:         # wait for ESC key to exit
                                break

                    sys.stderr.write("inference on {}:  {} / {}\r".format(video.name, idx+1, len(video)))

                toc /= cv2.getTickFrequency()
                # save results
                video_path = os.path.join(args.result_path, args.dataset, model_name,
                        'baseline', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_{:03d}.txt'.format(video.name, cnt+1))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        if isinstance(x, int):
                            f.write("{:d}\n".format(x))
                        else:
                            f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
                log = '({:3d}) Video ({:2d}): {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                        v_idx+1, cnt+1, video.name, toc, idx / toc, lost_number)
                print(log)
                with open(os.path.join(args.result_path, args.dataset, model_name, 'log.txt'), 'a') as f:
                    f.write(log + '\n')
                video_total_lost += lost_number
            total_lost += video_total_lost
            if args.repetition > 1:
                log = '({:3d}) Video: {:12s} Avg Lost: {:.3f}'.format(v_idx+1, video.name, video_total_lost/args.repetition)
                print(log)
                with open(os.path.join(args.result_path, args.dataset, model_name, 'log.txt'), 'a') as f:
                    f.write(log + '\n')

        log = "{:s} total (avg) lost: {:.3f}".format(model_name, total_lost/args.repetition)
        print(log)
        with open(os.path.join(args.result_path, args.dataset, model_name, 'log.txt'), 'a') as f:
            f.write(log + '\n')
    else:
        # OPE tracking

        find_best = True
        if args.dataset in ['TrackingNet', 'GOT-10k']:
            find_best = False


        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue

            best_pred_bboxes = []
            min_lost_number = 1e6
            for cnt in range(args.repetition):
                toc = 0
                pred_bboxes = []
                track_times = []
                template_image = None
                search_image = None
                prev_search_image = None
                raw_heatmap = None
                post_heatmap = None
                lost_number = 0

                if find_best and min_lost_number < args.min_lost_rate_for_repeat * len(video):
                    print("Abolish reset of trails ({}~) becuase the min lost number is small enough: {} / {}".format(cnt+1 , min_lost_number, args.min_lost_rate_for_repeat * len(video)))
                    break

                for idx, (img, gt_bbox) in enumerate(video):
                    tic = cv2.getTickCount()
                    if idx == 0:
                        outputs = tracker.init(img, gt_bbox)
                        pred_bbox = gt_bbox
                        pred_bboxes.append(pred_bbox)
                    else:
                        outputs = tracker.track(img)
                        pred_bbox_ = outputs['bbox']
                        pred_bbox = [pred_bbox_[0], pred_bbox_[1],
                                     pred_bbox_[2] - pred_bbox_[0],
                                     pred_bbox_[3] - pred_bbox_[1]]
                        pred_bboxes.append(pred_bbox)

                    toc += cv2.getTickCount() - tic
                    track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                    if idx == 0:
                        if args.vis:
                            cv2.destroyAllWindows()
                            if args.debug_vis:
                                for key, value in outputs.items():
                                    if isinstance(value, np.ndarray):
                                        if len(value.shape) == 3 or len(value.shape) == 2:
                                            cv2.imshow(key, value)
                    else:
                        if not gt_bbox == [0,0,0,0] and not np.isnan(np.array(gt_bbox)).any():
                            if pred_bbox[0] + pred_bbox[2] < gt_bbox[0] or pred_bbox[0] > gt_bbox[0] + gt_bbox[2] or pred_bbox[1] + pred_bbox[3] < gt_bbox[1] or pred_bbox[1] > gt_bbox[1] + gt_bbox[3]:
                                lost_number += 1

                        if find_best and lost_number > min_lost_number:
                            break


                        if args.vis or args.debug_vis:
                            gt_bbox = list(map(lambda x: int(x) if not np.isnan(x) else 0, gt_bbox))
                            pred_bbox = list(map(int, pred_bbox))
                            cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                          (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                            cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                          (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                            cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            cv2.imshow(video.name, img)

                            if args.debug_vis:

                                if prev_search_image is not None:
                                    cv2.imshow('prev_search_image', prev_search_image)

                                for key, value in outputs.items():
                                    if isinstance(value, np.ndarray):
                                        if len(value.shape) == 3 or len(value.shape) == 2:
                                            cv2.imshow(key, value)

                                            if key == 'search_image':
                                                prev_search_image = value

                                k = cv2.waitKey(0)
                                if k == 27:         # wait for ESC key to exit
                                    min_lost_number = 1e6 # this allows to  try args.repetition times for debug
                                    lost_number = 1e6 # this allows to  try args.repetition times for debug
                                    break
                            else:
                                k = cv2.waitKey(1)
                                if k == 27:         # wait for ESC key to exit
                                    min_lost_number = 1e6 # this allows to  try args.repetition times for debug
                                    lost_number = 1e6 # this allows to  try args.repetition times for debug
                                    break

                    sys.stderr.write("inference on {}:  {} / {}\r".format(video.name, idx+1, len(video)))

                if find_best and lost_number > min_lost_number:
                    print('Stop No.{} trial becuase the lost number already exceed the min lost number: {} > {} '.format(cnt+1, lost_number, min_lost_number))
                    continue

                if lost_number == 1e6:
                    continue

                if lost_number < min_lost_number:
                    min_lost_number = lost_number

                toc /= cv2.getTickFrequency()
                # save results
                if 'GOT-10k' == args.dataset:
                    video_path = os.path.join(args.result_path, args.dataset, model_name, video.name)
                    if not os.path.isdir(video_path):
                        os.makedirs(video_path)
                    result_path = os.path.join(video_path, '{}_{:03d}.txt'.format(video.name, cnt+1))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([vot_float2str("%.4f", i) for i in x ])+'\n')
                    result_path = os.path.join(video_path,
                            '{}_time.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in track_times:
                            f.write("{:.6f}\n".format(x))
                else:
                    model_path = os.path.join(args.result_path, args.dataset, model_name)
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path)
                    result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')

                log = '({:3d}) Video: {:12s} Trail: {:2d}  Time: {:5.1f}s Speed: {:3.1f}fps Lost: {:d}/{:d}'.format(
                    v_idx+1, video.name, cnt+1, toc, idx / toc, lost_number, len(video))
                print(log)
                with open(os.path.join(args.result_path, args.dataset, model_name, 'log.txt'), 'a') as f:
                    f.write(log + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Benchmark dataset inference', parents=[get_args_parser()])
    args = parser.parse_args()

    # create tracker
    if args.use_baseline_tracker:
        tracker = build_baseline_tracker(args)
    else:
        tracker = build_online_tracker(args)

    main(args, tracker)
