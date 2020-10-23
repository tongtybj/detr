########################################################################
# Download and decode YouTube BoundingBox Video
########################################################################

from __future__ import unicode_literals
from subprocess import check_call
from concurrent import futures
from datetime import datetime
import subprocess
import youtube_dl
import os
import sys
import cv2
import numpy as np
from yt_utils import parse_annotations, find_nearest_index, d_sets

# Debug flag. Set this to true if you only want to download single video and docode it
debug = False

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

      labeled_timestamps = np.array(clip.timestamps) / 1000

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
          frame_path = class_dir+'/'+ clip.yt_id +'_'+str(clip.timestamps[i])+\
              '_'+str(clip.class_id)+'_'+str(clip.obj_id)+'.jpg'
          cv2.imwrite(frame_path, image)
          #print(frame_path)
    capture.release()

  # Remove the temporary video
  if not debug:
    os.remove(d_set_dir+'/'+vid.yt_id+'_temp.mp4')


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

