
import json
import os

from tqdm import tqdm

from .dataset import Dataset
from .video import Video
from glob import glob

class GOT10kVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, gt_rects, img_names, load_img=False):
        super(GOT10kVideo, self).__init__(name, root, name,
                gt_rects[0], img_names, gt_rects, None, load_img)


class GOT10kDataset(Dataset):
    """
    Args:
        name:  dataset name, should be 'GOT-10k'
        dataset_root, dataset root dir
        single_video: a sinlge video from dataset
    """
    def __init__(self, name, dataset_root, load_img=False, single_video=None):
        super(GOT10kDataset, self).__init__(name, dataset_root)

        # load videos
        with open(os.path.join(dataset_root, 'list.txt'), 'r') as f:
            video_names = f.read().splitlines()

        pbar = tqdm(video_names, desc='loading '+name, ncols=100)

        self.videos = {}
        for video in pbar:

            if single_video and single_video != video:
                continue

            if not os.path.isdir(os.path.join(dataset_root, video)):
                continue

            pbar.set_postfix_str(video)

            img_names = [os.path.join(video, os.path.basename(x)) for x in sorted(glob(os.path.join(dataset_root, video, '*.jpg')), key=lambda x:int(os.path.basename(x).split('.')[0]))]
            gt_rects = [[0,0,0,0]] * len(img_names)
            with open(os.path.join(dataset_root, video, 'groundtruth.txt'), 'r') as f:
                gt_rects[0] = list(map(float, f.readline().strip().split(',')))

            self.videos[video] = GOT10kVideo(video, dataset_root, gt_rects, img_names)

        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
