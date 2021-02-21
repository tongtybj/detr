# Testing dataset directory
## Benchmarks
-  [VOT](http://www.votchallenge.net/)
-  [OTB](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html)
-  [UAV123](https://ivul.kaust.edu.sa/Pages/Dataset-UAV123.aspx)
-  [NFS](http://ci2cv.net/nfs/index.html)
-  [LaSOT](https://cis.temple.edu/lasot/)
-  [TrackingNet (Evaluation on Server)](https://tracking-net.org)
-  [GOT-10k (Evaluation on Server)](http://got-10k.aitestunion.com)

## Download Dataset

### VOT
Please read README.md in `benchmark/VOT`

### OTB (OTB100)

Please read README.md in `benchmark/OTB`

### UAV (UAV123)

Please read README.md in `benchmark/UAV`

### NFS (NFS30)

Please read README.md in `benchmark/NFS`

### TrackingNet

- Please read README.md in `benchmark/TrackingNet`
- Please submit the tracking result as `.zip` file to the organization. Please follow [the official instruction](https://github.com/SilvioGiancola/TrackingNet-devkit)

### LaSOT
- Please **manually** download the test dataset from https://drive.google.com/file/d/1EpeWYN4Li7eTvzTYg-B917S7RWNbwzHv/view
- Please **manually** unzip the dataset
- Please **manually** create a symbolic link: `benchmark/dataset/LaSOT`
- Please submit the tracking result as `.zip` file to the organization. Please follow [the official instruction](http://got-10k.aitestunion.com/submit_instructions)

### GOT-10K
- Please **manually** download the test dataset `Test data only` from http://got-10k.aitestunion.com/downloads
- Please **manually** unzip the dataset
- Please **manually** create a symbolic link: `benchmark/dataset/GOT-10k`: `ln -s ${PWD}/test GOT-10k`
