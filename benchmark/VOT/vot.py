import sys
import copy
import collections
import os
import argparse
import torch
import numpy as np
import cv2

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from benchmark.test import get_args_parser
from models import build_model
from models.hybrid_tracker import build_tracker

try:
    import trax
except ImportError:
    raise Exception('TraX support not found. Please add trax module to Python path.')

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])

def convert_vot_anno_to_rect(vot_anno, type):
    if len(vot_anno) == 4:
        return vot_anno

    if type == 'union':
        x1 = min(vot_anno[0::2])
        x2 = max(vot_anno[0::2])
        y1 = min(vot_anno[1::2])
        y2 = max(vot_anno[1::2])
        return [x1, y1, x2 - x1, y2 - y1]
    elif type == 'preserve_area':
        if len(vot_anno) != 8:
            raise ValueError

        vot_anno = np.array(vot_anno)
        cx = np.mean(vot_anno[0::2])
        cy = np.mean(vot_anno[1::2])

        x1 = min(vot_anno[0::2])
        x2 = max(vot_anno[0::2])
        y1 = min(vot_anno[1::2])
        y2 = max(vot_anno[1::2])

        A1 = np.linalg.norm(vot_anno[0:2] - vot_anno[2: 4]) * np.linalg.norm(vot_anno[2: 4] - vot_anno[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1

        x = cx - 0.5*w
        y = cy - 0.5*h
        return [x, y, w, h]
    else:
        raise ValueError


def run_vot(checkpoint):

    parser = argparse.ArgumentParser('TRTR model', parents=[get_args_parser()])
    args = parser.parse_args()
    args.checkpoint = checkpoint

    # hard-coding
    args.return_layers = ['layer3']
    args.enc_layers = 1
    args.dec_layers = 1

    # create tracker
    tracker = build_tracker(args)


    def _convert_anno_to_list(vot_anno):
        vot_anno = [vot_anno[0][0][0], vot_anno[0][0][1], vot_anno[0][1][0], vot_anno[0][1][1],
                    vot_anno[0][2][0], vot_anno[0][2][1], vot_anno[0][3][0], vot_anno[0][3][1]]
        return vot_anno

    def _convert_image_path(image_path):
        image_path_new = image_path[20:- 2]
        return "".join(image_path_new)

    """Run tracker on VOT."""
    handle = VOT("polygon")

    vot_anno_polygon = handle.region()
    vot_anno_polygon = _convert_anno_to_list(vot_anno_polygon)

    init_state = convert_vot_anno_to_rect(vot_anno_polygon, 'preserve_area')

    image_path = handle.frame()
    if not image_path:
        return
    image_path = _convert_image_path(image_path)
    image = cv2.imread(image_path)

    tracker.init(image, init_state)

    # Track
    while True:
        image_path = handle.frame()
        if not image_path:
            break
        image_path = _convert_image_path(image_path)

        image = cv2.imread(image_path)
        out = tracker.track(image)
        state = out['bbox']

        handle.report(Rectangle(state[0], state[1], state[2] - state[0], state[3] - state[1]))

class VOT(object):
    """ Base class for Python VOT integration """
    def __init__(self, region_format, channels=None):
        """ Constructor

        Args:
            region_format: Region format options
        """
        assert(region_format in [trax.Region.RECTANGLE, trax.Region.POLYGON])

        if channels is None:
            channels = ['color']
        elif channels == 'rgbd':
            channels = ['color', 'depth']
        elif channels == 'rgbt':
            channels = ['color', 'ir']
        elif channels == 'ir':
            channels = ['ir']
        else:
            raise Exception('Illegal configuration {}.'.format(channels))

        self._trax = trax.Server([region_format], [trax.Image.PATH], channels)

        request = self._trax.wait()
        assert(request.type == 'initialize')
        if isinstance(request.region, trax.Polygon):
            self._region = Polygon([Point(x[0], x[1]) for x in request.region])
        else:
            self._region = Rectangle(*request.region.bounds())
        self._image = [str(x) for k, x in request.image.items()]
        if len(self._image) == 1:
            self._image = self._image[0]
        self._trax.status(request.region)

    def region(self):
        """
        Send configuration message to the client and receive the initialization
        region and the path of the first image

        Returns:
            initialization region
        """

        return self._region

    def report(self, region, confidence = None):
        """
        Report the tracking results to the client

        Arguments:
            region: region for the frame
        """
        assert(isinstance(region, Rectangle) or isinstance(region, Polygon))
        if isinstance(region, Polygon):
            tregion = trax.Polygon.create([(x.x, x.y) for x in region.points])
        else:
            tregion = trax.Rectangle.create(region.x, region.y, region.width, region.height)
        properties = {}
        if not confidence is None:
            properties['confidence'] = confidence
        self._trax.status(tregion, properties)

    def frame(self):
        """
        Get a frame (image path) from client

        Returns:
            absolute path of the image
        """
        if hasattr(self, "_image"):
            image = self._image
            del self._image
            return tuple(image)

        request = self._trax.wait()

        if request.type == 'frame':
            image = [str(x) for k, x in request.image.items()]
            if len(image) == 1:
                image = image[0]
            return tuple(image)
        else:
            return None


    def quit(self):
        if hasattr(self, '_trax'):
            self._trax.quit()

    def __del__(self):
        self.quit()

