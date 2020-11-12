# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets.dataset import build as build_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from models.tracker import build_tracker
from benchmark import eval as benchmark_eval
from benchmark import test as benchmark_test

'''
usage:
$ python main.py --dataset_paths ./datasets/yt_bb/youtube_bb/Curation ./datasets/vid/ILSVRC2015/Curation/ --dataset_video_frame_ranges 100 3 --daset_num_uses 100000 -1 # check pysot
'''

def get_args_parser():
    parser = argparse.ArgumentParser('tracking evaluation', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_false',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)") #defualt is true
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
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--decoder_query', default=16, type=int) # hard-coding, should be obtained from the backbone calculation with the search_size
    parser.add_argument('--return_layers', default=[], nargs='+')
    parser.add_argument('--weighted', action='store_false',
                        help="the weighted for the multiple input embedding for transformer")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Loss coefficients
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_paths', default=[], nargs='+') # the path to datasets

    parser.add_argument('--dataset_video_frame_ranges', default=[100], nargs='+')
    parser.add_argument('--dataset_num_uses', default=[-1], nargs='+')
    parser.add_argument('--template_aug_shift', default=4, type=int)
    parser.add_argument('--template_aug_scale', default=0.05, type=float)
    parser.add_argument('--template_aug_color', default=1.0, type=float) # Pysot is 1.0
    parser.add_argument('--search_aug_shift', default=64, type=int)
    parser.add_argument('--search_aug_scale', default=0.18, type=float)
    parser.add_argument('--search_aug_color', default=1.0, type=float)  # Pysot is 1.0
    parser.add_argument('--exempler_size', default=127, type=int)
    parser.add_argument('--search_size', default=255, type=int)
    parser.add_argument('--negative_aug_rate', default=0.8, type=float)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
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

    parser.add_argument('--model_save_step', default=50, type=int,
                        help='step to save model')
    parser.add_argument('--benchmark_test_step', default=2, type=int,
                        help='step to test benchmark')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print("args: {}".format(args))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    benchmark_test_parser = argparse.ArgumentParser('benchmark dataset inference', parents=[benchmark_test.get_args_parser(), get_args_parser()],conflict_handler='resolve')
    benchmark_test_args = benchmark_test_parser.parse_args()
    benchmark_test_args.result_path = Path(os.path.join(args.output_dir, 'benchmark'))
    benchmark_test_args.dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'benchmark')

    benchmark_eval_parser = argparse.ArgumentParser('benchmark dataset inference', parents=[benchmark_eval.get_args_parser(), get_args_parser()],conflict_handler='resolve')
    benchmark_eval_args = benchmark_eval_parser.parse_args()
    benchmark_eval_args.tracker_path = benchmark_test_args.result_path
    best_eao = 0
    best_eao_epoch = 0

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # training
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every args.model_save_step epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.model_save_step == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

            # hack: only inference model
            utils.save_on_master({'model': model_without_ddp.state_dict()}, output_dir / 'checkpoint_only_inference.pth')

        # evalute
        val_stats = evaluate(model, criterion, postprocessors, data_loader_val, device, args.output_dir)

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


        # evualute with benchmark
        if utils.is_main_process():
            if (epoch + 1) % args.benchmark_test_step == 0:
                model.eval()
                tracker = build_tracker(model, postprocessors["bbox"], benchmark_test_args)
                benchmark_start_time = time.time()
                benchmark_test.main(benchmark_test_args, tracker)
                benchmark_time = time.time() - benchmark_start_time

                eval_results = benchmark_eval.main(benchmark_eval_args)
                eval_result = list(eval_results.values())[0]

                if benchmark_test_args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
                    if args.output_dir:
                        with (output_dir / str("benchmark_" +  benchmark_test_args.dataset + ".txt")).open("a") as f:
                            f.write("epoch: " + str(epoch) + ", " + json.dumps(eval_result) + ", best EAO: " + str(best_eao) +  "\n")

                    if best_eao < eval_result['EAO']:

                        if args.output_dir:
                            # remove the old one:
                            best_eao_int = int(best_eao*1000)
                            checkpoint_path = output_dir / f'checkpoint{best_eao_epoch:04}_best_eao_{best_eao_int:03}.pth'
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)
                            checkpoint_path = output_dir / f'checkpoint{best_eao_epoch:04}_best_eao_{best_eao_int:03}_only_inference.pth'
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)


                        best_eao = eval_result['EAO']
                        best_eao_epoch = epoch

                        if args.output_dir:
                            best_eao_int = int(best_eao*1000)
                            checkpoint_path = output_dir / f'checkpoint{epoch:04}_best_eao_{best_eao_int:03}.pth'
                            utils.save_on_master({
                                'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'args': args,
                            }, checkpoint_path)

                            # hack: only inference model
                            utils.save_on_master({'model': model_without_ddp.state_dict()}, output_dir / f'checkpoint{epoch:04}_best_eao_{best_eao_int:03}_only_inference.pth')


                print("benchmark time: {}".format(benchmark_time))
        if args.distributed:
            torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
