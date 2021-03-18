**TrTr**: Tracker with Transformer

# install

```
$ ./install.sh ~/anaconda3 trtr 
```
**note**: anaconda3 path
**note**: please select a proper cudatoolkit version to install pytorch from conda, the default is 10.1. For RTX3090, please select 11.0. Then the command would be  `$ ./install.sh ~/anaconda3 trtr 11.0`

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

## Benchmark

### Bacis usage
```
$ python test.py --cfg_file ../parameters/experiment/vot2018/offline.yaml  --result_path yaml_test --model_name test
```


### Hyper-parameter search
```
$ python hp_search.py --tracker.checkpoint ../networks/trtr_resnet50.pth --tracker.search_sizes 280 --separate --repetition 1  --use_baseline_tracker --tracker.model.transformer_mask True
```


# Train

## Test with single GPU
```
$ python main.py  --cfg_file ./parameters/train/default.yaml  --batch_size 16 --dataset.paths ./datasets/yt_bb/dataset/Curation  ./datasets/vid/dataset/Curation/ --dataset.video_frame_ranges 3 100  --dataset.num_uses 100 100  --dataset.eval_num_uses 100 100  --benchmark_start_epoch 0   --resume networks/trtr_resnet50.pth --output_dir temp --epochs 10
```
**note**: maybe you have to modify the file limit: `ulimit -n 8192`. Write in `~/.bashrc` maybe better

## Multi-GPU

### multi-GPU in single machine


### multi-GPU in multi machine