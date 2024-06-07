# TinyYolo #

If you like tiny code, Pytorch and Yolo, then you'll like TinyYolo.

## What this is ##

* This repo uses the new "Tiny-Oriented-Programming" paradigm invented by [TinyGrad](https://github.com/tinygrad/tinygrad) to implement a set of popular Yolo models.

* No YAML files, just (hopefully) good, modular, readable, minimal yet complete code.

## What this isn't ##

* This isn't a framework. If you want a fine-tuned framework then use darknet or ultralytics

* A highly optimized library (I haven't implement conv+bn fusion, or anything like that yet)

## What's provided ##

* `models.py`: contains all the Yolo models, which automatically calculate loss when targets are provided in forward function.
* `test.py`: tests the models with pretrained weights from darknet, ultralytics and Yolov7.
* `train_coco.py`: trains on COCO using [lightning](https://lightning.ai/)

## Observations ##

* Yolov7 uses `eps=0.001` and `momentum=0.03` in `nn.Batchnorm2d`. That's unusual. I wonder what effects that has on training.