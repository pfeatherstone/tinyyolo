# TinyYolo #

If you like tiny code, Pytorch and Yolo, then you'll like TinyYolo.

## What this is ##

* This repo uses the new "Tiny-Oriented-Programming" paradigm invented by [TinyGrad](https://github.com/tinygrad/tinygrad) to implement a set of popular Yolo models and sample assignment algorithms.

* No YAML files, just (hopefully) good, modular, readable, minimal yet complete code.

* This is a library, not a framework.

* This library is for developers more so than for users.

## What's provided ##

* `models.py`: contains all the Yolo models, which automatically calculate loss when targets are provided in forward function.
* `assigner.cpp`: contains various sample assignment algorithms written in pure Pytorch C++
* `test.py`: tests the models with pretrained weights from darknet, ultralytics and Yolov7.
* `train_coco.py`: an example training script that trains on COCO using [lightning](https://lightning.ai/)

## Models ##

- [x] Yolov3
- [x] Yolov3-spp
- [x] Yolov3-tiny
- [x] Yolov4
- [x] Yolov4-tiny
- [x] Yolov5(n,s,m,l,x)
- [x] Yolov7
- [x] Yolov8(n,s,m,l,x)
- [x] Yolov10(n,s,m,b,l,x)

## Assigners ##
- [x] [FCOS](https://arxiv.org/pdf/1904.01355)
- [x] [ATSS](https://arxiv.org/pdf/1912.02424)
- [x] [TAL](https://arxiv.org/pdf/2108.07755)

## Notes ##

- I advise using [lightning](https://lightning.ai/) or [accelerate](https://huggingface.co/docs/accelerate/index) to write training scripts. They take care of everything including distributed training, FP16, checkpointing, etc.

- The sample assignment algorithms are written in Pytorch C++ for simplicity. Indeed, when I first wrote them in Python, 90% of the complexity was in crazy tensor indexing and masking. This was destracting and annoying. In C++ you can use for-loops and if-statements without loss of performance. The algorithms are much more readable now. The only drawback is that you have to put tensors back onto CPU, perform the algorithm, then put the returned tensors (target boxes, scores and classes) back onto target device, usually GPU.

## Observations ##

* All the ultralytics models and Yolov7 use `eps=0.001` and `momentum=0.03` in `nn.Batchnorm2d`. That's unusual. I wonder what effects that has on training.

## TODO ##

- [ ] Train everything (probably going to need some cloud compute (help))
- [ ] API docs + examples