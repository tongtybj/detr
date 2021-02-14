from os.path import join, isdir
from os import listdir, mkdir, makedirs
import sys
import cv2
import numpy as np
import os
import pickle
import json
import glob
from concurrent import futures
import xml.etree.ElementTree as ET
import argparse

sys.path.append('..')
import utils

base_path = "./dataset"
save_base_path = join(base_path, 'Curation')
TRAIN_SUBSET_PREFIX = "TRAIN_"
# TEST_SUBSET_PREFIX = "TEST" # no test dataset for val


def crop_video(subdir, subset, video):
    video_crop_base_path = join(save_base_path, subset, video)
    if not isdir(video_crop_base_path):
        makedirs(video_crop_base_path)
    else:
        print(video_crop_base_path, " is already curated")
        return

    #print("crop video: {}".format(video))
    frames = glob.glob(join(subdir, 'videos', video, '*.jpg'))
    def index_keys(text):
        return int(text.split('/')[-1].split('.')[0])
    frames = sorted(frames, key=index_keys)

    with open(join(subdir, 'anno', video + '.txt')) as f:
        ann = f.readlines()

    for id, frame in enumerate(frames):

        im = cv2.imread(frame)
        avg_chans = np.mean(im, axis=(0, 1))
        bbox_str = [float(s) for s in  ann[id].split(',')]

        trackid = 0
        bbox = [bbox_str[0], bbox_str[1], bbox_str[0]+bbox_str[2], bbox_str[1]+bbox_str[3]]

        filename = frame.split('/')[-1].split('.')[0]
        z, x = utils.crop_image(im, bbox, padding=avg_chans)
        cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(filename), trackid)), x)


def image_curation(num_threads, subset, video_num):

    if subset == -1:
        subset_dirs = sorted(glob.glob(join(base_path, TRAIN_SUBSET_PREFIX + '*') + "[!zip]"))
        #subset_dirs.append(join(base_path, TEST_SUBSET_PREFIX))
    else:
        subset_dirs = [join(base_path, TRAIN_SUBSET_PREFIX + str(subset))]

    for subdir  in subset_dirs:

        videos = sorted(os.listdir((join(subdir, 'videos'))))

        if video_num > 0:
            videos = videos[:video_num]

        n_videos = len(videos)

        subset = subdir.split('/')[-1]

        # without multiprocess
        #for video in videos:
        #    crop_video(subdir, subset, video)

        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_video, subdir, subset, video) for video in videos]
            for i, f in enumerate(futures.as_completed(fs)):
                utils.print_progress(i, n_videos, prefix=subset, suffix='Done ', barLength=40)

def save_config():
    snippets = dict()
    instance_size = utils.get_instance_size()

    subset_dirs = sorted(glob.glob(join(save_base_path, TRAIN_SUBSET_PREFIX + '*')))

    # since the data size is too large, we recommand to curate subset one by one.
    # then first reload an existing annotation file.
    if os.path.isfile(join(save_base_path, 'train.pickle')):
        with open(join(save_base_path, 'train.pickle'), 'rb') as f:
            snippets = pickle.load(f)
            print("existing annotation file contains {} videos ".format(len(snippets.keys())))


    for subdir in subset_dirs:
        subset = subdir.split('/')[-1]

        videos = sorted(os.listdir(subdir))
        # print(videos)
        print(subdir)
        for vi, video in enumerate(videos):

            if join(subset, video) in snippets:
                # print('already have {}/{} video id: {:04d} / {:04d}'.format(subdir, video, vi, len(videos)))
                continue

            sys.stdout.write('\r subset: {} video id: {:04d} / {:04d}'.format(subdir, vi, len(videos)))
            sys.stdout.flush()

            frames = glob.glob(join(base_path, subset, 'videos', video, '*.jpg'))
            # print(subdir, video, frames)
            assert len(frames) > 0

            def index_keys(text):
                return int(text.split('/')[-1].split('.')[0])
            frames = sorted(frames, key=index_keys)

            with open(join(base_path, subset, 'anno', video + '.txt')) as f:
                ann = f.readlines()

            frame_sz = None
            snippets[join(subset, video)] = dict()
            snippets[join(subset, video)]['tracks'] = dict()
            snippet = dict()
            for id, frame in enumerate(frames):
                if id == 0:
                    im = cv2.imread(frame)
                    frame_sz = [im.shape[1], im.shape[0]]
                    snippets[join(subset, video)]['frame_size'] = frame_sz

                bbox_str = [float(s) for s in  ann[id].split(',')]
                bbox = [bbox_str[0], bbox_str[1], bbox_str[0]+bbox_str[2], bbox_str[1]+bbox_str[3]]

                if bbox[0] < 0:
                    bbox[0] = 0
                if bbox[1] < 0:
                    bbox[1] = 0

                filename = frame.split('/')[-1].split('.')[0]
                snippet['{:06d}'.format(int(filename))] = bbox
            snippets[join(subset, video)]['tracks']['00'] = snippet


    train = snippets
    val = {k:v for i, (k,v) in enumerate(snippets.items()) if i < 100}

    print('save annotation files')
    # json file is for debug
    json.dump(train, open(join(save_base_path,'train.json'), 'w'), indent=4, sort_keys=True)
    json.dump(val, open(join(save_base_path,'val.json'), 'w'), indent=4, sort_keys=True)

    with open(join(save_base_path, 'train.pickle'), 'wb') as f:
        pickle.dump(train, f)

    with open(join(save_base_path, 'val.pickle'), 'wb') as f:
        pickle.dump(val, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('curate trackingnet', add_help=False)
    parser.add_argument('--num_thread', default=12, type=int)
    parser.add_argument('--subset', default=-1, type=int)
    parser.add_argument('--video_num', default=-1, type=int)
    args = parser.parse_args()

    if not isdir(save_base_path): mkdir(save_base_path)

    print("crop the images for training")

    image_curation(args.num_thread, args.subset, args.video_num)

    print("save the configuration for the curation data")

    save_config()
