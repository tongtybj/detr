from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import sys
import cv2
import torch
import numpy as np
from glob import glob
from models.tracker import build_tracker as build_baseline_tracker
from models.hybrid_tracker import build_tracker as build_online_tracker
from models.hybrid_tracker import get_args_parser as tracker_args_parser

def get_args_parser():
    parser = argparse.ArgumentParser('demo', add_help=False, parents=[tracker_args_parser()])

    parser.add_argument('--use_baseline_tracker', action='store_true')
    parser.add_argument('--video_name', default="", type=str)
    parser.add_argument('--debug', action='store_true', help='wheterh visualize the debug result')

    return parser


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        ext = os.listdir(video_name)[0].split(".")[-1]
        assert ext == "jpg" or ext == "JPEG" or ext == "png"

        images = glob(os.path.join(video_name, '*.' + ext))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main(args, tracker):


    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    template_image = None
    for frame in get_frames(args.video_name):

        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()

            tracker.init(frame, init_rect)
            first_frame = False
            continue


        output = tracker.track(frame)

        bbox = np.round(output["bbox"]).astype(np.uint16)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (0, 255, 0), 3)
        cv2.imshow("result", frame)

        wait = 1
        if args.debug:
            wait = 0
            for key, value in output.items():
                if isinstance(value, np.ndarray):
                    if len(value.shape) == 3 or len(value.shape) == 2:
                        cv2.imshow(key, value)

        k = cv2.waitKey(wait)

        if k == 27:         # wait for ESC key to exit
            sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # create tracker
    if args.use_baseline_tracker:
        tracker = build_baseline_tracker(args)
    else:
        tracker = build_online_tracker(args)

    main(args, tracker)

