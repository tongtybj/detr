# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import sys
import cv2
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
    parser.add_argument('--encoder_feedforward', action='store_true')
    parser.add_argument('--decoder_feedforward', action='store_false') # TODO: not good
    parser.add_argument('--return_layers', default=[], nargs='+')
    parser.add_argument('--weighted', action='store_true',
                        help="the weighted for the multiple input embedding for transformer")
    parser.add_argument('--transformer_mask', action='store_false',
                        help="mask for transformer") # workaround to enable transformer mask for default
    parser.add_argument('--multi_frame', action='store_true',
                        help="use multi frame for encoder (template images)")

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
    parser.add_argument('--tracking_size_lpf', default=0.33, type=float,
                        help='the factor of the lpf for size tracking')

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
                                            load_img=False)

    model_name = args.model_name

    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []

            template_image = None
            search_image = None
            raw_heatmap = None
            post_heatmap = None

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
                            cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 2))], True, (0, 255, 0), 3)

                            bbox = list(map(int, pred_bbox))
                            cv2.rectangle(img, (bbox[0], bbox[1]),
                                          (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                            cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            cv2.putText(img, 'lost', (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.imshow(video.name, img)

                            debug_images = ['template_image', 'prev_template_image', 'search_image', 'raw_heatmap', 'post_heatmap', 'atom_heatmap']
                            for image in debug_images:
                                if image in outputs and outputs[image] is not None:
                                    cv2.imshow(image, outputs[image])

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

                        debug_images = ['template_image', 'prev_template_image', 'search_image', 'raw_heatmap', 'post_heatmap', 'atom_heatmap']
                        for image in debug_images:
                            if image in outputs and outputs[image] is not None:
                                cv2.imshow(image, outputs[image])
                        k = cv2.waitKey(0)
                        if k == 27:         # wait for ESC key to exit
                            sys.exit()
                    else:
                        cv2.waitKey(1)
                elif not args.vis:
                    sys.stderr.write("inference on {}:  {} / {}\r".format(video.name, idx+1, len(video)))

            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join(args.result_path, args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            template_image = None
            search_image = None
            raw_heatmap = None
            post_heatmap = None

            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    tracker.init(img, gt_bbox)
                    pred_bbox = gt_bbox
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox_ = outputs['bbox']
                    pred_bbox = [pred_bbox_[0], pred_bbox_[1],
                                 pred_bbox_[2] - pred_bbox_[0],
                                 pred_bbox_[3] - pred_bbox_[1]]
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['score'])

                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    if args.vis:
                        cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)

                    if args.debug_vis:
                        debug_images = ['template_image', 'prev_template_image', 'search_image', 'raw_heatmap', 'post_heatmap', 'atom_heatmap']
                        for image in debug_images:
                            if image in outputs and outputs[image] is not None:
                                cv2.imshow(image, outputs[image])

                        k = cv2.waitKey(0)
                        if k == 27:         # wait for ESC key to exit
                            sys.exit()
                    else:
                        cv2.waitKey(1)
                else:
                    sys.stderr.write("inference on {}:  {} / {}\r".format(video.name, idx+1, len(video)))

            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join(args.result_path, args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join(args.result_path, args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
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
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Benchmark dataset inference', parents=[get_args_parser()])
    args = parser.parse_args()

    # create tracker
    if args.use_baseline_tracker:
        tracker = build_baseline_tracker(args)
    else:
        tracker = build_online_tracker(args)

    main(args, tracker)
