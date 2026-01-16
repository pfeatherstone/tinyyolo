import pytest
import numpy as np
import torch
from   torch.export import Dim
import onnxruntime as ort
from   models import *


def get_models():
    yield Yolov3(80, False)
    yield Yolov3(80, True)
    yield Yolov3Tiny(80)
    yield Yolov4(80)
    yield Yolov4Tiny(80)
    yield Yolov7(80)
    yield from [Yolov6(v, 80) for v in "nsm"]
    yield from [Yolov5(v, 80)  for v in "nsmlx"]
    yield from [Yolov8(v, 80)  for v in "nsmlx"]
    yield from [Yolov10(v, 80) for v in "nsmblx"]
    yield from [Yolov11(v, 80) for v in "nsmlx"]
    yield from [Yolov12(v, 80) for v in "nsmlx"]


@pytest.mark.parametrize("net", get_models())
@torch.inference_mode()
def test_forward(net: YoloBase):
    net     = net.eval()
    x       = torch.randn(4, 3, 512, 768)
    B,H,W   = x.shape[0], x.shape[2], x.shape[3]
    N       = sum([(H//s)*(W//s) for s in net.head.strides])
    nd      = N*3 if isinstance(net.head, DetectV3) else N
    nc      = 85 if isinstance(net.head, DetectV3) else 84
    out     = net(x)
    assert out.shape[0] == B, "whoops net {}".format(type(net))
    assert out.shape[1] == nd, "whoops net {}".format(type(net))
    assert out.shape[2] == nc, "whoops net {}".format(type(net))


@pytest.mark.parametrize("net", get_models())
@torch.inference_mode()
def test_export(net: YoloBase):
    net = net.eval()
    x = torch.randn(4, 3, 640, 640)
    _ = net(x) # compile einops kernels just in case
    torch.onnx.export(net, (x,), dynamo=True, opset_version=23,
                      input_names=['img'],
                      output_names=['preds'],
                      dynamic_shapes={'x' : (Dim.DYNAMIC, Dim.STATIC, Dim.DYNAMIC, Dim.DYNAMIC)}).save('/tmp/model.onnx')
    netOrt  = ort.InferenceSession('/tmp/model.onnx', providers=['CPUExecutionProvider'])
    x       = torch.randn(2, 3, 576, 768)
    preds1  = net(x) 
    preds2, = netOrt.run(None, {'img': x.numpy()})
    torch.testing.assert_close(preds1[...,:4], torch.from_numpy(preds2[...,:4]), atol=1e-3, rtol=1e-2) # boxes
    torch.testing.assert_close(preds1[...,4:], torch.from_numpy(preds2[...,4:]), atol=5e-5, rtol=1e-4) # scores
