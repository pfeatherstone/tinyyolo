# TinyYolo #

If you like tiny code, Pytorch and Yolo, then you'll like TinyYolo.

## What this is ##

* This repo uses the new "Tiny-Oriented-Programming" paradigm invented by [TinyGrad](https://github.com/tinygrad/tinygrad) to implement a set of popular Yolo models.

* No YAML files, just (hopefully) good, modular, readable, minimal yet complete code.

## What's provided ##

* `models.py`: contains all the Yolo models, which automatically calculate loss when targets are provided in forward function.
* `test.py`: tests the models with pretrained weights from darknet, ultralytics and Yolov7.
* `train_coco.py`: trains on COCO using [lightning](https://lightning.ai/)

## Models ##

- [x] Yolov3
- [x] Yolov3-spp
- [x] Yolov3-tiny
- [x] Yolov4
- [x] Yolov4-tiny
- [x] Yolov5(n,s,m,l,x)
- [x] Yolov7
- [x] YOlov8(n,s,m,l,x)

## Observations ##

* Yolov7 uses `eps=0.001` and `momentum=0.03` in `nn.Batchnorm2d`. That's unusual. I wonder what effects that has on training.