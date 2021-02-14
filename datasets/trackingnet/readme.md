# Preprocessing COCO

### Download dataset

- the dataset size is too large to use console comand (e.g., gdown) to downlaod. Currently, manual downloading from web browser is the only way to get the dataset. Please download dataset from https://drive.google.com/drive/folders/1gJOR-r-jPFFFCzKKlMOW80WFtuaMiaf6 (all chunks)

- there are 12 subsets (TRAIN_0 ~ TRAIN_11), where each subst contrain 2500 videos, so the total video is about 25000

- the quality 

````shell
ln -sfb $PWD/TrackingNet ./dataset # important
````

### Unzip the files
````shell
python unzip.py
````
#### options:
- `--subset`: assign a spefic subset (i.e., 0~11), default is `-1` meaning unzip all subsets
- `--video_num`: only unzip number of `video_num` inside one subset (i.e., one subset contain 2500 videos). 
- sample: `python unzip.py --subset 0 --video_num 10 # unzip TRAIN_0 within 10 vidoes` 

### Visualize
````shell
python visualize.py
````

### Crop & Generate data info (10 min)

````shell
python curate.py
````

#### options 
- `--num_trhead`: the numver of thread for parallel processing
- `--subset`: assign a spefic subset (i.e., 0~11), default `-1` means unzip all subsets
- `--video_num`: only unzip number of `video_num` inside one subset (i.e., one subset contain 2500 videos), default `-1` means curate all videos in each subset
- sample: `python curate.py --subset 0 --video_num 20` 
