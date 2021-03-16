**TrTr**: Tracker with Transformer

# install

```
$ ./install.sh ~/anaconda3 trtr
```
**note**: anaconda3 path

```
conda activate trtr
```

# Demo

## Offline
```
$ python demo.py --checkpoint networks/trtr_resnet50.pth --return_layers layer3 --use_baseline_tracker
```

### image sequences
add `--video_name ${video_dir}`

### video
add `--video_name ${video_name}`


## Online
```
$ python demo.py --checkpoint networks/trtr_resnet50.pth --return_layers layer3 --dcf_layers layer2 layer3 --dcf_rate 0.6
```




# Train

## Test with single GPU
```
$ python main.py --batch_size 16 --dataset_paths ./datasets/yt_bb/dataset/Curation  ./datasets/vid/dataset/Curation/ --dataset_video_frame_ranges 3 100  --dataset_num_uses 100 100 --return_layers layer3 --enc_layers 1 --dec_layers 1  --eval_dataset_num_uses 100 100
```

## Multi-GPU

### multi-GPU in single machine


### multi-GPU in multi machine