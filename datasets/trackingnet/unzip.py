from os.path import join, isdir
from os import listdir, mkdir, makedirs
import sys
import cv2
import numpy as np
import glob
from concurrent import futures
import zipfile
import os
import argparse

sys.path.append('..')
import utils

base_path = "./dataset"
SUBSET_PREFIX = "TRAIN_"

def unzip_subsets(subset):

    if subset == -1:
        subset_dirs = sorted(glob.glob(join(base_path, SUBSET_PREFIX + '*zip')))
    else:
        subset_dirs = [join(base_path, SUBSET_PREFIX + str(subset) + '.zip')]

    #print(subset_dirs)

    for subset_dir in subset_dirs:
        print('unzip {}'.format(subset_dir))
        subset_name = subset_dir.split('/')[-1].split('.')[0]
        unzip_path = join(base_path, subset_name)
        if not isdir(unzip_path):
            mkdir(unzip_path)
            with zipfile.ZipFile(subset_dir, 'r') as zf:
                zf.extractall(unzip_path)
            print('finish unzip ', subset_name)
        else:
            print(subset_name, ' is already unzipped!')

def unzip_videos(subset, video_num):

    if subset == -1:
        subset_dirs = sorted(glob.glob(join(base_path, SUBSET_PREFIX + '*') + "[!zip]"))
    else:
        subset_dirs = [join(base_path, SUBSET_PREFIX + str(subset))]

    #print(subset_dirs)

    for subset_dir in subset_dirs:
        save_base_path = join(subset_dir, 'videos')
        if not isdir(save_base_path): mkdir(save_base_path)

        zip_videos = sorted(glob.glob(join(subset_dir, 'zips', '*.zip')))

        print('{} has {} zipped videos'.format(subset_dir, len(zip_videos)))

        if video_num > 0:
            zip_videos= zip_videos[:video_num]

        for id, zip_video in enumerate(zip_videos):
            #print(zip_video)
            video_name = zip_video.split('/')[-1].split('.')[0]
            save_path = join(save_base_path, video_name)
            if not isdir(save_path):
                mkdir(save_path)
                with zipfile.ZipFile(zip_video, 'r') as zf:
                    zf.extractall(save_path)
                    sys.stdout.write('\r unzip {}/{}: {:04d} / {:04d}'.format(subset_dir.split('/')[-1], video_name, id+1, len(zip_videos)))
                    sys.stdout.flush()

            else:
                print(video_name, ' is already unzipped!')
            os.remove(zip_video)

        if video_num == -1:
            os.rmdir(join(subset_dir, 'zips'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('unzip trackingnet', add_help=False)
    parser.add_argument('--subset', default=-1, type=int)
    parser.add_argument('--video_num', default=-1, type=int)
    args = parser.parse_args()
    unzip_subsets(args.subset)
    unzip_videos(args.subset, args.video_num)
