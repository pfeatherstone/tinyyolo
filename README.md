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
- [ ] Yolov11(n,s,m,l,x)
- [ ] Yolov12(n,s,m,l,x)

## Assigners ##
- [x] [FCOS](https://arxiv.org/pdf/1904.01355)
- [x] [ATSS](https://arxiv.org/pdf/1912.02424)
- [x] [TAL](https://arxiv.org/pdf/2108.07755)

## Example ##

```
nc  = ... # number of classes, e.g. 80 for COCO
net = Yolov3(nc, spp=True).eval()
# net = Yolov4(nc).eval()
# net = Yolov5('n', nc).eval()
# net = Yolov6('n', nc).eval()
# net = Yolov7(nc).eval()
# net = Yolov8('n', nc).eval()
# net = Yolov10('n', nc).eval()

# Inference only
B = ... # Batch size
H = ... # Height
W = ... # Width
x = torch.randn(B, 3, H, W) # image-like
preds = net(x) # preds.shape == [B, N, nc+5] or nc+4 if there's no objectness feature e.g V5, V6, V8, V10

# Train
D = ... # max number of target detections per batch
C = 5   # target features [x1,y1,x2,y2,cls] where x1,x2 ∈ [0,W], y1,y2 ∈ [0,H], cls ∈ [-1,nc) and -1 is used for padding
targets = torch.randn(B, D, C)
preds, loss_dict = net(x, targets)

```

## Export ##

### ONNX ###

Export it...

```
net = Yolov3(80, spp=True).eval()

x = torch.randn(4, 3, 640, 640)
_ = net(x) # compile all the einops kernels. Required before ONNX export
torch.onnx.export(net, (x,), '/tmp/model.onnx',
                  input_names=['img'], output_names=['preds'],
                  dynamic_axes={'img'   : {0: 'B', 2: 'H', 3: 'W'},
                                'preds' : {0: 'B', 1: 'N'}})
```

Run it...

Install dependencies:

```
pip install numpy onnxruntime
```

```
import onnxruntime as ort
import numpy as np

net    = ort.InferenceSession('/tmp/model.onnx', providers=['CPUExecutionProvider'])
x      = np.random.randn((1, 3, 576, 768))
preds, = net.run(None, {'img': x})
```

Compile it...

Download onnxmlir and use the [onnx-mlir.py](https://github.com/onnx/onnx-mlir/blob/main/docs/Docker.md#easy-script-to-compile-a-model) script.

```
python3 onnx-mlir.py --EmitObj -O3 /tmp/model.onnx -o model 
```

### TFLite ###

Convert it...

Install dependencies:

```
pip install onnx2tf tensorflow tf_keras onnx_graphsurgeon sng4onnx onnxsim
```

```
onnx2tf -i /tmp/model.onnx -ois "img:1,3,640,640" -o /tmp/model
```

## Notes ##

- I advise using [lightning](https://lightning.ai/) or [accelerate](https://huggingface.co/docs/accelerate/index) to write training scripts. They take care of everything including distributed training, FP16, checkpointing, etc.

- The sample assignment algorithms are written in Pytorch C++ for simplicity. Indeed, when I first wrote them in Python, 90% of the complexity was in crazy tensor indexing and masking. This was destracting and annoying. In C++ you can use for-loops and if-statements without loss of performance. The algorithms are much more readable now. The only drawback is that you have to put tensors back onto CPU, perform the algorithm, then put the returned tensors (target boxes, scores and classes) back onto target device, usually GPU.

## Observations ##

* Pretty much all official models use `eps=0.001` and `momentum=0.03` in `nn.Batchnorm2d`. Those aren't the Pytorch defaults. Where do those numbers come from?

* From what I can tell the main innovation in yolov6 is the distillation loss in bounding box regression: there are two branches for bounding box, one with DFL and one without. AFAIK, only the DFL one gets used in forward pass. During training, both get CIOU loss-ed.

* onnx-mlir is very slow and runs on 1 thread only. So onnxruntime is better for inferrence.

## TODO ##

- [ ] Train everything (probably going to need some cloud compute (help))
- [ ] Train with mixed precision
- [ ] Maybe add Yolov9