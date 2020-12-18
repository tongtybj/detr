# Preprocessing COCO

### Download dataset

Please download dataset from http://got-10k.aitestunion.com/downloads_dataset/full_data and unzip


````shell
ln -sfb $PWD/got10k ./dataset # important
````

### Crop & Generate data info (10 min)

````shell
python curate.py
````
