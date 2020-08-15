

This repo contains modified Ultralytics inference and training code for YOLOv3 in PyTorch. The code works on Linux, MacOS and Windows. Credit to Joseph Redmon for YOLO  https://pjreddie.com/darknet/yolo/.


## Requirements

Python 3.7 or later with all `requirements.txt` dependencies installed, including `torch >= 1.5`. To install run:
```bash
$ pip install -U -r requirements.txt
```

## How to use
* download weights from google drive link in weights folder
* look at "simple_test.ipynb" to test if all dependencies are installed correctly and to see the simplest way to pass the model on images is
* note: classes.names is in cfg folder


## Data Info
* approximately 1068 images 1080x1920
* 56,047 individual worms

## Citation

[![DOI](https://zenodo.org/badge/146165888.svg)](https://zenodo.org/badge/latestdoi/146165888)
