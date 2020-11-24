# Preprocessing COCO

### Download raw images and annotations

````shell
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip ./train2017.zip
unzip ./val2017.zip
unzip ./annotations_trainval2017.zip

ln -sfb $PWD/coco ./dataset # important
````

### Crop & Generate data info (10 min)

````shell
#python curate.py [option: num_threads]
python curate.py
````
