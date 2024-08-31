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
* `test.py`: tests the models with pretrained weights from darknet, ultralytics, Yolov6 and Yolov7 official repos.
* `train_coco.py`: an example training script that trains on COCO using [lightning](https://lightning.ai/)

## Models ##

- [x] Yolov3
- [x] Yolov3-spp
- [x] Yolov3-tiny
- [x] Yolov4
- [x] Yolov4-tiny
- [x] Yolov5(n,s,m,l,x)
- [x] Yolov6(n,s,m)
- [x] Yolov7
- [x] Yolov8(n,s,m,l,x)
- [x] Yolov10(n,s,m,b,l,x)

## Assigners ##
- [x] [FCOS](https://arxiv.org/pdf/1904.01355)
- [x] [ATSS](https://arxiv.org/pdf/1912.02424)
- [x] [TAL](https://arxiv.org/pdf/2108.07755)

## Example ##

```
net = Yolov3(80, spp=True).eval()
# net = Yolov4(80).eval()
# net = Yolov5('n', 80).eval()
# net = Yolov6('n', 80).eval()
# net = Yolov7(80).eval()
# net = Yolov8('n', 80).eval()
# net = Yolov10('n', 80).eval()

# Inference only
x = torch.randn(4, 3, 640, 640) # image-like
preds = net(x) # preds.shape == [4, N, 85] or 84 if there's no objectness feature e.g V5, V6, V8, V10

# Train
D = ... # max number of target detections per batch
C = 5   # target features [x1,y1,x2,y2,cls] where cls == -1 is used for padding
targets = torch.randn(4, D, C)
preds, loss_dict = net(x, targets)

```

## Export ##

### ONNX ###

```
net = Yolov3(80, spp=True).eval()
# net = Yolov4(80).eval()
# net = Yolov5('n', 80).eval()
# net = Yolov6('n', 80).eval()
# net = Yolov7(80).eval()
# net = Yolov8('n', 80).eval()
# net = Yolov10('n', 80).eval()

x = torch.randn(4, 3, 640, 640)
_ = net(x) # compile all the einops kernels. Required before ONNX export
torch.onnx.export(net, (x,), '/tmp/model.onnx',
                  input_names=['img'], output_names=['preds'],
                  dynamic_axes={'img'   : {0: 'B', 2: 'H', 3: 'W'},
                                'preds' : {0: 'B', 1: 'N'}})
```

## Notes ##

- I advise using [lightning](https://lightning.ai/) or [accelerate](https://huggingface.co/docs/accelerate/index) to write training scripts. They take care of everything including distributed training, FP16, checkpointing, etc.

- The sample assignment algorithms are written in Pytorch C++ for simplicity. Indeed, when I first wrote them in Python, 90% of the complexity was in crazy tensor indexing and masking. This was destracting and annoying. In C++ you can use for-loops and if-statements without loss of performance. The algorithms are much more readable now. The only drawback is that you have to put tensors back onto CPU, perform the algorithm, then put the returned tensors (target boxes, scores and classes) back onto target device, usually GPU.

## Observations ##

* Pretty much all official models use `eps=0.001` and `momentum=0.03` in `nn.Batchnorm2d`. Those aren't the Pytorch defaults. Where do those numbers come from?

* From what I can tell the main innovation in yolov6 is the distillation loss in bounding box regression: there are two branches for bounding box, one with DFL and one without. AFAIK, only the DFL one gets used in forward pass. During training, both get CIOU loss-ed.

## TODO ##

- [ ] Train everything (probably going to need some cloud compute (help))
- [ ] Train with mixed precision
- [ ] API docs + examples (in README)
- [ ] Add examples for ONNX export, TFLITE export.
- [ ] Explore how to compile models. Try TVM, onnx-mlir, TinyGrad (export to ONNX, load into tinygrad, then export LLVM or C code then compile)
- [ ] Maybe add Yolov9