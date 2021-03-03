# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import time
import datetime
import test as benchmark
import itertools
from models.tracker import build_tracker as build_baseline_tracker
from models.hybrid_tracker import build_tracker as build_online_tracker


def get_args_parser(parser, hpnames):

    for name in hpnames:
        parser.add_argument('--' + name, default=[], nargs='+')

    parser.add_argument('--separate_mode', action='store_true',
                        help="best score mode if separate_mode is false")

    parser.add_argument('--save_path', default='hp_search', type=str)
    
    return parser

def main(args, hpnames):

    # TODO: record model configuration in checkpoint
    if len(args.return_layers) == 0:
        args.return_layers = ['layer3']

    if len(args.dcf_layers) == 0:
        args.dcf_layers = ['layer2', 'layer3']

    # separate mode or best score mode
    repetition = 1
    if args.separate_mode:
        repetition = args.repetition
        args.repetition  = 1 # separate model (non best score model)
        

    # create dir for result
    layers_info = 'trtr_layer'
    for layer in args.return_layers:
        layers_info += '_' + layer[-1]
    if args.use_baseline_tracker:
        layers_info += '_baseline'
    else:
        layers_info += '_dcf_layer'
        for layer in args.dcf_layers:
            layers_info += '_' + layer[-1]


    if args.separate_mode:
        args.save_path += '_separate'
    args.result_path = os.path.join(args.save_path,
                                    os.path.splitext(os.path.basename(args.checkpoint))[0],
                                    layers_info)
    dataset_path = os.path.join(args.result_path, args.dataset)
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)


    hparams = dict()
    for name in hpnames:
        val = getattr(args, name)
        core_name = name[:-1]
        default_value = getattr(args, core_name)
        assert core_name in vars(args).keys()

        if len(val) == 0:
            hparams[name] = [default_value]
        else:
            hparams[name] = [type(default_value)(v) for v in val]
    hparams['run'] = list(range(1, repetition + 1))

    tracker_num = len(list(itertools.product(*hparams.values())))

    for tracker_id, hparam_set in enumerate(itertools.product(*hparams.values())):
        t = time.time()
        print("start {}/{} tracker test".format(tracker_id + 1, tracker_num))
        model_name = ''
        if args.use_baseline_tracker:
            model_name = 'baseline_'
        for idx, (name, val) in enumerate(zip(hparams.keys(), hparam_set)):
            if hasattr(args, name):
                name = name[:-1] # core name
                setattr(args, name, val)

            if args.use_baseline_tracker and 'dcf' in name:
                continue

            model_name += name + "_" + str(val).replace('.', 'p')
            if idx < len(hparam_set) - 1:
                model_name += '_'

        if not args.use_baseline_tracker:
            model_name += '_false_positive' # workaround to distinguish with old model name

        #print(model_name)
        model_dir = os.path.join(dataset_path, model_name)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        args.model_name = model_name
        with open(os.path.join(model_dir, 'log.txt'), 'a') as f:
            f.write('parameters: \n')
            f.write('{}'.format(vars(args)) + '\n\n')

        # create tracker
        if args.use_baseline_tracker:
            tracker = build_baseline_tracker(args)
        else:
            tracker = build_online_tracker(args)

        # start test with benchmark
        benchmark.main(args, tracker)

        du = round(time.time() - t)
        print("finish {}/{} tracker test, take {}, rest {} ".format(tracker_id + 1, tracker_num, datetime.timedelta(seconds = du), datetime.timedelta(seconds = du * (tracker_num - tracker_id - 1))))

if __name__ == '__main__':

    hpnames = ['search_sizes', 'window_factors', 'tracking_size_lpfs', 'dcf_sizes', 'dcf_rates', 'dcf_sample_memory_sizes'] # please add hyperparameter here

    #hpnames = ['search_sizes']


    parser = argparse.ArgumentParser('hyperparameter search for benchmark', parents=[get_args_parser(benchmark.get_args_parser(), hpnames)])
    args = parser.parse_args()

    #print(args)

    main(args, hpnames)

