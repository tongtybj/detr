import os
import json

from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video

class UAVVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        img_names: image names
        gt_rect: groundtruth rectangle

    """
    def __init__(self, name, root, gt_rects, img_names, load_img=False):
        super(UAVVideo, self).__init__(name, root, name, gt_rects[0], img_names, gt_rects, None, load_img)

class UAVDataset(Dataset):
    """
    Args:
        name: dataset name, should be UAV (only suppot UAV123)
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False, single_video=None):
        super(UAVDataset, self).__init__(name, dataset_root)

        # load videos
        dataset_dir = os.path.join(dataset_root, 'dataset', 'UAV123')
        anno_files = glob(os.path.join(dataset_dir, 'anno', 'UAV123', '*.txt'))
        assert len(anno_files) == 123
        video_names = [x.split('/')[-1].split('.')[0] for x in anno_files]

        pbar = tqdm(video_names, desc='loading '+name, ncols=100)

        self.videos = {}
        for idx, video in enumerate(pbar):

            if single_video and single_video != video:
                continue

            video_dir = os.path.join(dataset_dir, 'data_seq', 'UAV123', video)
            if not os.path.isdir(video_dir):
                continue

            img_names = sorted(glob(os.path.join(video_dir, '*.jpg')), key=lambda x:int(os.path.basename(x).split('.')[0]))

            assert anno_files[idx].split('/')[-1].split('.')[0] == video

            with open(anno_files[idx], 'r') as f:
                gt_rects = [list(map(float, x.strip().split(','))) for x in f.readlines()]

            if len(img_names) > len(gt_rects):
                img_names = img_names[0:len(gt_rects)]


            pbar.set_postfix_str(video)
            self.videos[video] = UAVVideo(video, dataset_root, gt_rects, img_names)

        # set attr
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
