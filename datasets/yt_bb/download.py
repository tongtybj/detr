########################################################################
# YouTube BoundingBox
########################################################################
#
# This file contains useful functions for downloading, decoding, and
# converting the YouTube BoundingBox dataset.
#
########################################################################

from __future__ import unicode_literals
from subprocess import check_call
from concurrent import futures
from random import shuffle
from datetime import datetime
import time
import subprocess
import youtube_dl
import socket
import os
import io
import sys
import csv
import cv2
import numpy as np


# Debug flag. Set this to true if you would like to see ffmpeg errors
debug = False

# The data sets to be downloaded
d_sets = [
  'yt_bb_detection_train',
  'yt_bb_detection_validation'
]

# Host location of segment lists
web_host = 'https://research.google.com/youtube-bb/'

# Video clip class
class video_clip(object):
  def __init__(self,
               name,
               yt_id,
               start,
               class_id,
               obj_id,
               d_set_dir):
    # name = yt_id+class_id+object_id
    self.name     = name
    self.yt_id    = yt_id
    self.frames = [start]
    self.class_id = class_id
    self.obj_id   = obj_id
    self.d_set_dir = d_set_dir
  def print_all(self):
    print('['+self.name+', '+ \
              self.yt_id+', '+ \
              self.frames+', '+ \
              self.class_id+', '+ \
              self.obj_id+']\n')

# Video class
class video(object):
  def __init__(self,yt_id,first_clip):
    self.yt_id = yt_id
    self.clips = [first_clip]
  def print_all(self):
    print(self.yt_id)
    for clip in self.clips:
      clip.print_all()


# Help function to get the index of the element in an array the nearest to a value
def find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


# Download and cut a clip to size
def dl_and_cut(vid):

  d_set_dir = vid.clips[0].d_set_dir

  # Use youtube_dl to download the video
  FNULL = open(os.devnull, 'w')
  check_call(['youtube-dl', \
    #'--no-progress', \
    '-f','best[ext=mp4]', \
    '-o',d_set_dir+'/'+vid.yt_id+'_temp.mp4', \
    'youtu.be/'+vid.yt_id ], \
     stdout=FNULL,stderr=subprocess.STDOUT )

  video_path = d_set_dir+'/'+vid.yt_id+'_temp.mp4'

  # Verify that the video has been downloaded. Skip otherwise
  if os.path.exists(video_path):

    # Use opencv to open the video
    capture = cv2.VideoCapture(video_path)
    fps, total_f = capture.get(5), capture.get(7)

    #print("total_f: {}, fps: {}".format(total_f, fps))
    # Get time stamps (in seconds) for every frame in the video
    # This is necessary because some video from YouTube come at 30/29.99/24 fps
    timestamps = np.array([i/float(fps) for i in range(int(total_f))])
    #print("video_path: {}, timestamep: {}, vid clips: {}".format(vid.yt_id, timestamps, len(vid.clips)))

    for clip in vid.clips:

      labeled_timestamps = np.array(clip.frames) / 1000

      indexes = []
      for label in labeled_timestamps:
        frame_index = find_nearest_index(timestamps, label)
        indexes.append(frame_index)

      #print("clip: {}, label ts {}, match index: {}".format(clip.name, labeled_timestamps, indexes))

      # Make the class directory if it doesn't exist yet
      class_dir = d_set_dir+'/'+str(clip.class_id)
      check_call(' '.join(['mkdir', '-p', class_dir]), shell=True)

      for i, index in enumerate(indexes):
          # Get the actual image corresponding to the frame
          capture.set(1, index)
          ret, image = capture.read()

          # Save the extracted image
          frame_path = class_dir+'/'+ clip.yt_id +'_'+str(clip.frames[i])+\
              '_'+str(clip.class_id)+'_'+str(clip.obj_id)+'.jpg'
          cv2.imwrite(frame_path, image)
          #print(frame_path)
    capture.release()

  # Remove the temporary video
  if not debug:
    os.remove(d_set_dir+'/'+vid.yt_id+'_temp.mp4')

# Parse the annotation csv file and schedule downloads and cuts
def parse_annotations(d_set,dl_dir):

  d_set_dir = dl_dir+'/'+d_set+'/'

  # Download & extract the annotation list
  if not os.path.exists(d_set+'.csv'):
    print (d_set+': Downloading annotations...')
    check_call(' '.join(['wget', web_host+d_set+'.csv.gz']),shell=True)
    print (d_set+': Unzipping annotations...')
    check_call(' '.join(['gzip', '-d', '-f', d_set+'.csv.gz']), shell=True)

  print (d_set+': Parsing annotations into clip data...')

  # Parse csv data.
  annotations = []
  with open((d_set+'.csv'), 'rt') as f:
    reader = csv.reader(f)
    annotations = list(reader)

  # Sort to de-interleave the annotations for easier parsing. We use
  # `int(l[1])` to sort by the timestamps numerically; the other fields are
  # sorted lexicographically as strings.
  print(d_set + ': Sorting annotations...')

  # Sort by youtube_id, class, obj_id and then timestamp
  annotations.sort(key=lambda l: (l[0], l[2], l[4], int(l[1])))

  current_clip_name = ['blank']
  clips             = []

  # Parse annotations into list of clips with names, youtube ids, start
  # times and stop times
  for idx, annotation in enumerate(annotations):
    yt_id    = annotation[0]
    class_id = annotation[2]
    obj_id   = annotation[4]

    clip_name = yt_id+'+'+class_id+'+'+obj_id

    # If this is a new clip
    if clip_name != current_clip_name:

      #if idx != 0:
      #  print(clips[-1].name, clips[-1].frames)

      # Add the starting clip
      clips.append( video_clip(
        clip_name,
        yt_id,
        int(annotation[1]),
        class_id,
        obj_id,
        d_set_dir) )

      # Update the current clip name
      current_clip_name = clip_name

    else:
      clips[-1].frames.append(int(annotation[1]))

  # Sort the clips by youtube id
  clips.sort(key=lambda x: x.yt_id)

  # Create list of videos to download (possibility of multiple clips
  # from one video)
  current_vid_id = ['blank']
  vids = []
  for clip in clips:

    vid_id = clip.yt_id

    # If this is a new video
    if vid_id != current_vid_id:
      # Add the new video with its first clip
      vids.append( video ( \
        clip.yt_id, \
        clip ) )
    # If this is a new clip for the same video
    else:
      # Add the new clip to the video
      vids[-1].clips.append(clip)

    # Update the current video name
    current_vid_id = vid_id

  return annotations,clips,vids

# Parse the annotation csv file and schedule downloads and cuts
def parse_and_sched(dl_dir='videos',num_threads=4):
  """Download the entire youtube-bb data set into `dl_dir`.
  """

  # Make the download directory if it doesn't already exist
  check_call(['mkdir', '-p', dl_dir])

  # For each of the four datasets
  for d_set in d_sets:
    annotations,clips,vids = parse_annotations(d_set,dl_dir)

    d_set_dir = dl_dir+'/'+d_set+'/'

    # Make the directory for this dataset
    check_call(' '.join(['mkdir', '-p', d_set_dir]), shell=True)

    # Tell the user when downloads were started
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if debug: # only test with one video:
      vids = vids[:1]

    # Download and cut in parallel threads giving
    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
      fs = [executor.submit(dl_and_cut,vid) for vid in vids]
      for i, f in enumerate(futures.as_completed(fs)):
        # Write progress to error so that it can be seen
        sys.stderr.write( \
          "Downloaded and converted video: {} / {} \r".format(i, len(vids)))

    print( d_set+': All videos downloaded' )



if __name__ == '__main__':

  assert(len(sys.argv) == 3), \
          "Usage: python download.py [VIDEO_DIR] [NUM_THREADS]"
  # Use the directory `videos` in the current working directory by
  # default, or a directory specified on the command line.
  parse_and_sched(sys.argv[1],int(sys.argv[2]))

