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

    return parser

def main(args, hpnames):

    # TODO: record model configuration in checkpoint
    if len(args.return_layers) == 0:
        args.return_layers = ['layer3']

    if len(args.dcf_layers) == 0:
        args.dcf_layers = ['layer2', 'layer3']

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


    args.result_path = os.path.join('hp_search',
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


    tracker_num = len(list(itertools.product(*hparams.values())))

    for tracker_id, hparam_set in enumerate(itertools.product(*hparams.values())):
        t = time.time()
        print("start {}/{} tracker test".format(tracker_id + 1, tracker_num))
        model_name = ''
        for idx, (name, val) in enumerate(zip(hparams.keys(), hparam_set)):
            name = name[:-1] # core name
            assert hasattr(args, name)
            setattr(args, name, val)

            model_name += name + "_" + str(val).replace('.', 'p')
            if idx < len(hparam_set) - 1:
                model_name += '_'

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

