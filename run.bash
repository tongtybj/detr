#!/usr/bin/env bash

#--userns=host: https://dockerlabs.collabnix.com/advanced/security/userns/

docker run --name trtr-train \
       --gpus all \
       --mount type=bind,src=${2:-~/detr},dst=/root/detr \
       --mount type=bind,src=${1:-~}/datasets,dst=/root/datasets \
       --shm-size=4g \
       -it --privileged \
       --network=host \
       --userns=host \
       trtr
