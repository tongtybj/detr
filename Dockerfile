FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev libgl1-mesa-dev && \
    rm -rf /var/cache/apk/*

RUN pip --no-cache-dir install Cython

COPY requirements.txt /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt

# hack for dlbox 
RUN pip install torch==1.5.1+cu92 torchvision==0.6.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# install vot-toolkit-python
RUN pip install git+https://github.com/votchallenge/vot-toolkit-python
