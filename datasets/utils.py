import sys
import cv2
import numpy as np

test_flag = False

# Note: if you have a exemplar size with 127 and a search size with 255, then you need instance size of 511 for later data augmentation in dataset
INSTANCE_SIZE = 511
EXEMPLAR_SIZE = 127
CONTEXT_AMOUNT= 0.5

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


def siamfc_like_scale(bbox, crop_size):

    bb_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]

    wc_z = bb_size[1] + CONTEXT_AMOUNT * sum(bb_size)
    hc_z = bb_size[0] + CONTEXT_AMOUNT * sum(bb_size)

    s_z = np.sqrt(wc_z * hc_z)
    scale_z = crop_size / s_z

    return s_z, scale_z

def crop_image(image, bbox, padding=(0, 0, 0)):


    def pos_s_2_bbox(pos, s):
        return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


    bb_center = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    s_z =  siamfc_like_scale(bbox, EXEMPLAR_SIZE)[0]
    s_x = INSTANCE_SIZE / EXEMPLAR_SIZE  * s_z
    #print("INSTANCE_SIZE: {}; EXEMPLAR_SIZE: {}; s_z: {}; s_x: {}".format(INSTANCE_SIZE, EXEMPLAR_SIZE, s_z, s_x))

    z = crop_hwc(image, pos_s_2_bbox(bb_center, s_z), EXEMPLAR_SIZE, padding)
    x = crop_hwc(image, pos_s_2_bbox(bb_center, s_x), INSTANCE_SIZE, padding) # crop a size of s_x, then resize to INSTANCE_SIZE

    #z = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    if test_flag:

        rec_image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),3)
        cv2.imshow('raw_image', rec_image)
        cv2.imshow('z_image', z)
        cv2.imshow('x_image', x)

        k = cv2.waitKey(40)
        if k == 27:         # wait for ESC key to exit
          sys.exit()

    return z, x
