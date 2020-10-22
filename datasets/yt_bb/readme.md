# Preprocessing Youtube BoundingBox 
https://research.google.com/youtube-bb/download.html


### Download video and extract annotated frames (~400GB)
````shell
ln -sbf $PWD/youtube_bb ./ # we need a sufficient disk space to store the dataset, please be careful to set $PWD
python download.py ./youtube_bb 24 # argv[1]: path to store argv[2]: the worker number
````

### Crop & Generate data info (~ 1day?)
```shell
python curate_video.py
```
