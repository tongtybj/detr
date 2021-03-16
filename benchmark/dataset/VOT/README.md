# Prepare for VOT benchmark

## Create symbolic link for dataset  (Option, but recommended)

````shell
ln -sfb $PWD/dataset ./dataset
````
**note**: suppose you have sufficient space under `$PWD/dataset`

## Download and unzip dataset

````shell
./install.sh       # 2018 (default)
./install.sh 2020  # 2020
````

