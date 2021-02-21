# Prepare for VOT benchmark

## Download test dataset

Please **manually** download the test dataset `TEST.zip` from https://drive.google.com/drive/u/0/folders/1gJOR-r-jPFFFCzKKlMOW80WFtuaMiaf6

**note**: file size is too big, cannot use gdown

## Create directory for dataset

````shell
ln -sfb $PWD/dataset ./dataset # $PWD/dataset is the directoty contain TEST.zip
````

## unzip dataset

````shell
python unzip.py
````

