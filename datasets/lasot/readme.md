# Preprocessing COCO

### Download dataset

Please download dataset (all chunks) from https://drive.google.com/file/d/1O2DLxPP8M4Pn4-XCttCJUW3A29tDIeNa/view   and unzip
**note**: you can also download a single subset from https://drive.google.com/drive/folders/1v09JELSXM_v7u3dF7akuqqkVG8T1EK2_

````shell
ln -sfb $PWD/lasot ./dataset # important
````

### Crop & Generate data info (10 min)

````shell
python curate.py
````
