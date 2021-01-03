from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from models import build_model
from models.tracker import build_tracker

torch.set_num_threads(1)

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer tracking', add_help=False)

    # Model parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # * Backbone
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--resnet_dilation', action='store_false',
                        help="If true (default), we replace stride with dilation in resnet blocks") #default is true
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_false',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)") #defualt is true
    parser.add_argument('--return_interm_layers', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
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
    parser.add_argument('--weighted', action='store_true',
                        help="the weighted for the multiple input embedding for transformer")
    parser.add_argument('--transformer_mask', action='store_false',
                        help="mask for transformer") # workaround to enable transformer mask for defanult
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
    parser.add_argument('--video_name', default="", type=str)
    parser.add_argument('--checkpoint', default="", type=str)
    parser.add_argument('--exemplar_size', default=127, type=int)
    parser.add_argument('--search_size', default=255, type=int)
    parser.add_argument('--context_amount', default=0.5, type=float)

    # * hyper-parameter for tracking
    parser.add_argument('--score_threshold', default=0.1, type=float,
                        help='the lower score threshold to recognize a target (score_target > threshold) ')
    parser.add_argument('--window_steps', default=3, type=int,
                        help='the pyramid factor to gradually reduce the widow effect')
    parser.add_argument('--window_factor', default=0.4, type=float,
                        help='the factor of the hanning window for heatmap post process')
    parser.add_argument('--tracking_size_penalty_k', default=0.04, type=float,
                        help='the factor to penalize the change of size')
    parser.add_argument('--tracking_size_lpf', default=0.33, type=float,
                        help='the factor of the lpf for size tracking')

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


def main(args):
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available in Pytorch")

    device = torch.device('cuda')

    # create model
    assert args.transformer_mask # should be True
    model, criterion, postprocessors = build_model(args)

    # load model
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    assert 'model' in checkpoint
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    tracker = build_tracker(model, postprocessors["bbox"], args)

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


        output = tracker.track(frame)

        bbox = np.round(output["bbox"]).astype(np.uint16)
        print("the tracking score: {}, box: {}".format(output["score"], bbox))
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (0, 255, 0), 3)

        cv2.imshow("template", output["template_image"])
        cv2.imshow("search_raw", output["search_image"])
        cv2.imshow("raw_heatmap", output["raw_heatmap"])
        cv2.imshow("post_heatmap", output["post_heatmap"])
        cv2.imshow("result", frame)

        k = cv2.waitKey(0)
        #k = cv2.waitKey(40)
        if k == 27:         # wait for ESC key to exit
            sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
