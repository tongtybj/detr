# Preprocessing VID(Object detection from video)
Large Scale Visual Recognition Challenge 2015 (ILSVRC2015)

### Download dataset (86GB)

````shell
wget http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz
tar -xzvf ./ILSVRC2015_VID.tar.gz
ln -sfb $PWD/ILSVRC2015_./ # important
````

### Crop & Generate data info (20 min)

````shell
python curate_vid.py
````
**note**: you can add option like: `python curate_vid.py 511 24` 
