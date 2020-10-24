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
    parser.add_argument('--decoder_query', default=16, type=int) # hard-coding, should be obtained from the backbone calculation with the search_size

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=1, type=float,
                        help="Relative classification weight of the no-object class")

    # tracking
    parser.add_argument('--video_name', default="", type=str)
    parser.add_argument('--checkpoint', default="", type=str)
    parser.add_argument('--exemplar_size', default=127, type=int)
    parser.add_argument('--search_size', default=255, type=int)
    parser.add_argument('--context_amount', default=0.5, type=float)
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

        bbox = np.round(output["bbox"])
        print("the tracking class: {}, score: {}, box: {}".format(output["label"],  output["score"], bbox))
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (0, 255, 0), 3)

        cv2.imshow("template", output["template_image"])
        cv2.imshow("search_raw", output["search_image"])
        cv2.imshow("result", frame)

        #k = cv2.waitKey(0)
        k = cv2.waitKey(40)
        if k == 27:         # wait for ESC key to exit
            sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
