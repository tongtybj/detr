from os.path import join, isdir
from os import listdir, mkdir, makedirs
import sys
import cv2
import numpy as np
import pickle
import json
import glob
from multiprocessing.pool import ThreadPool
from concurrent import futures
import xml.etree.ElementTree as ET
import argparse

VID_base_path = "./ILSVRC2015"
save_base_path = join(VID_base_path, 'Curation')
ann_base_path = join(VID_base_path, 'Annotations/VID')
img_base_path = join(VID_base_path, 'Data/VID')

subdir_map = {'train/ILSVRC2015_VID_train_0000': 'a',
              'train/ILSVRC2015_VID_train_0001': 'b',
              'train/ILSVRC2015_VID_train_0002': 'c',
              'train/ILSVRC2015_VID_train_0003': 'd',
              'val': 'e'}

test_flag = False

def print_progress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def siamfc_like_scale(bbox, context_amount, crop_size):

    bb_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = bb_size[1] + context_amount * sum(bb_size)
    hc_z = bb_size[0] + context_amount * sum(bb_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = crop_size / s_z

    return s_z, scale_z

def crop_image(image, bbox, context_amount=0.5, exemplar_size=127, instance_size=255, padding=(0, 0, 0)):

    def pos_s_2_bbox(pos, s):
        return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]

    bb_center = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    s_z =  siamfc_like_scale(bbox, context_amount, exemplar_size)[0]
    s_x = instance_size / exemplar_size  * s_z
    #print("instance_size: {}; exemplar_size: {}; s_z: {}; s_x: {}".format(instance_size, exemplar_size, s_z, s_x))

    z = crop_hwc(image, pos_s_2_bbox(bb_center, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(bb_center, s_x), instance_size, padding) # crop a size of s_x, then resize to instance_size

    #z = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    if test_flag:

        rec_image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),3)
        cv2.imshow('raw_image', rec_image)
        cv2.imshow('z_image', z)
        cv2.imshow('x_image', x)
        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
          sys.exit()

    return z, x

def crop_video(subdir, sub_set, video, instance_size):
    video_crop_base_path = join(save_base_path, sub_set, video)
    if not isdir(video_crop_base_path): makedirs(video_crop_base_path)

    xmls = sorted(glob.glob(join(ann_base_path, subdir, video, '*.xml')))

    for xml in xmls:
        xmltree = ET.parse(xml)
        objects = xmltree.findall('object')
        objs = []
        filename = xmltree.findall('filename')[0].text

        im = cv2.imread(xml.replace('xml', 'JPEG').replace('Annotations', 'Data'))
        avg_chans = np.mean(im, axis=(0, 1))
        for object_iter in objects:
            trackid = int(object_iter.find('trackid').text)
            bndbox = object_iter.find('bndbox')

            bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
            z, x = crop_image(im, bbox, instance_size=instance_size, padding=avg_chans)

            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(int(filename), trackid)), z)
            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(filename), trackid)), x)



def image_curation(instance_size=511, num_threads=24):

    # (instanc_size + 1) / (search_size + 1) = (search_size + 1) / (exemplar_size + 1)

    for subdir, sub_set  in subdir_map.items():
        sub_set_base_path = join(ann_base_path, subdir)

        videos = sorted(listdir(sub_set_base_path))
        n_videos = len(videos)

        '''
        for video in videos:
            crop_video(subdir, sub_set, video, instance_size)
        '''

        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_video, subdir, sub_set, video, instance_size) for video in videos]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                print_progress(i, n_videos, prefix=sub_set, suffix='Done ', barLength=40)



def save_config():
    snippets = dict()

    for subdir, sub_set  in subdir_map.items():
        subdir_base_path = join(ann_base_path, subdir)
        videos = sorted(listdir(subdir_base_path))
        for vi, video in enumerate(videos):
            print('subset: {} video id: {:04d} / {:04d}'.format(subdir, vi, len(videos)))

            frames = []
            xmls = sorted(glob.glob(join(subdir_base_path, video, '*.xml')))

            id_set = []
            id_frames = [[]] * 60  # at most 60 objects

            for num, xml in enumerate(xmls):
                f = dict()
                xmltree = ET.parse(xml)
                size = xmltree.findall('size')[0]
                frame_sz = [int(it.text) for it in size]
                objects = xmltree.findall('object')
                objs = []
                for object_iter in objects:
                    trackid = int(object_iter.find('trackid').text)

                    name = (object_iter.find('name')).text
                    bndbox = object_iter.find('bndbox')
                    occluded = int(object_iter.find('occluded').text)
                    o = dict()
                    o['c'] = name
                    o['bbox'] = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                                 int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                    o['trackid'] = trackid
                    o['occ'] = occluded
                    objs.append(o)

                    if trackid not in id_set:
                        id_set.append(trackid)
                        id_frames[trackid] = []
                    id_frames[trackid].append(num)

                f['frame_sz'] = frame_sz
                f['img_path'] = xml.split('/')[-1].replace('xml', 'JPEG')
                f['objs'] = objs
                frames.append(f)

            if len(id_set) > 0:
                snippets[join(sub_set, video)] = dict()

            for selected in id_set:
                frame_ids = sorted(id_frames[selected])
                # check the following  behavior by np.split(np.array([0,2,3,5]), np.array(np.where(np.diff(np.array([0,2,3,5]))> 1)[0]) + 1)
                sequences = np.split(frame_ids, np.array(np.where(np.diff(frame_ids) > 1)[0]) + 1)
                sequences = [s for s in sequences if len(s) > 1]  # remove isolated frame.
                for seq in sequences:
                    snippet = dict()
                    for frame_id in seq:
                        frame = frames[frame_id]
                        for obj in frame['objs']:
                            if obj['trackid'] == selected:
                                o = obj
                                continue
                        snippet[frame['img_path'].split('.')[0]] = o['bbox']
                    snippets[join(sub_set, video)]['{:02d}'.format(selected)] = snippet

    train = {k:v for (k,v) in snippets.items() if 'train' in k}
    val = {k:v for (k,v) in snippets.items() if 'val' in k}


    #json.dump(train, open(join(save_base_path,'train.json'), 'w'), indent=4, sort_keys=True)
    #json.dump(val, open(join(save_base_path,'val.json'), 'w'), indent=4, sort_keys=True)

    with open(join(save_base_path, 'train.pickle'), 'wb') as f:
        pickle.dump(train, f)

    with open(join(save_base_path, 'val.pickle'), 'wb') as f:
        pickle.dump(val, f)


if __name__ == '__main__':

    if not isdir(save_base_path): mkdir(save_base_path)

    print("crop the images for training")

    # Note: if you have a exemplar size with 127 and a search size with 255, then you need instance size of 511 for later data augmentation in dataset
    if len(sys.argv) == 3:
        image_curation(sys.argv[1], sys.argv[2])
    else:
        image_curation()

    print("save the configuration for the curation data")

    save_config()
