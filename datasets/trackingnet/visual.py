from os.path import join, isdir
from os import listdir, mkdir, makedirs
import sys
import cv2
import numpy as np
import glob
import cv2
import os
import random


sys.path.append('..')
import utils

base_path = "./dataset"

def show_videos():

    subsets = glob.glob(join(base_path, 'TRAIN_*') + "[!zip]")
    #print(subsets)
    subset = random.choice(subsets)

    videos = sorted(os.listdir((join(subset, 'videos'))))
    # print(len(videos))
    print('select subset of ', subset.split('/')[-1])

    for video in videos:

        print(video)
        frames = glob.glob(join(subset, 'videos', video, '*.jpg'))
        def index_keys(text):
            return int(text.split('/')[-1].split('.')[0])
        frames = sorted(frames, key=index_keys)

        with open(join(subset, 'anno', video + '.txt')) as f:
            ann = f.readlines()

        for id, frame in enumerate(frames):
            im = cv2.imread(frame)

            bbox = [float(s) for s in  ann[id].split(',')]
            #print(bbox, frame)

            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
            cv2.rectangle(im, pt1, pt2, (0,255,0), 3)

            cv2.imshow('img', im)

            k = cv2.waitKey(20)
            if k == 27:         # wait for ESC key to exit
                sys.exit()



if __name__ == '__main__':

    show_videos()
