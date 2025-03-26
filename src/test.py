import  os
from    typing import Union
import  numpy as np
import  torch
import  torchvision
import  onnx
import  onnxruntime as ort
import  onnxslim
from    models import *

class bcolors:
    HEADER      = '\033[95m'
    OKBLUE      = '\033[94m'
    OKCYAN      = '\033[96m'
    OKGREEN     = '\033[92m'
    WARNING     = '\033[93m'
    FAIL        = '\033[91m'
    ENDC        = '\033[0m'
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'

weight_paths = {
    'yolov3-tiny'   : 'https://pjreddie.com/media/files/yolov3-tiny.weights',
    'yolov3'        : 'https://pjreddie.com/media/files/yolov3.weights',
    'yolov3-spp'    : 'https://github.com/ultralytics/yolov3/releases/download/v8/yolov3-spp.weights',
    'yolov4'        : 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights',
    'yolov4-tiny'   : 'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights',
}

def download_if_not_exist(model_type: str, filepath: str):
    if not os.path.exists(filepath):
        torch.hub.download_url_to_file(weight_paths[model_type], filepath)


def swap_convs(cv1, cv2):
    state1 = deepcopy(cv1.state_dict())
    state2 = deepcopy(cv2.state_dict())
    cv1.load_state_dict(state2)
    cv2.load_state_dict(state1)


def load_from_darknet(net: Union[Yolov3, Yolov3Tiny, Yolov4, Yolov4Tiny], weights_path: str):
    
    def params(net):
        # Handle special modules
        if isinstance(net, Yolov3):
            for module in [net.net,    net.fpn.b1, net.head.cv[2], 
                           net.fpn.c1, net.fpn.b2, net.head.cv[1], 
                           net.fpn.c2, net.fpn.b3, net.head.cv[0]]:
                yield from params(module)
        
        elif isinstance(net, Yolov3Tiny):
            for module in [net.net, net.fpn.b0, net.head.cv[1], 
                                    net.fpn.b1, net.head.cv[0]]:
                yield from params(module)
            
        elif isinstance(net, Yolov4):
            for module in [net.net, net.fpn.b1, net.fpn.c1, net.fpn.b2, 
                                    net.fpn.c2, net.fpn.b3, net.head.cv[0],
                                    net.fpn.c3, net.fpn.b4, net.head.cv[1],
                                    net.fpn.c4, net.fpn.b5, net.head.cv[2]]:
                yield from params(module)
        
        elif isinstance(net, Yolov4Tiny):
            for module in [net.net, net.fpn.b0, net.head.cv[1], net.fpn.b1, net.head.cv[0]]:
                yield from params(module)

        elif isinstance(net, BackboneV4):
            for module in net.children():
                yield from params(module)
        
        elif isinstance(net, C4):
            for module in [net.cv1, net.cv2, net.m, net.cv3, net.cv4]:
                yield from params(module)

        elif isinstance(net, Conv):
            yield from [net[1].bias, net[1].weight, net[1].running_mean, net[1].running_var, net[0].weight]
        
        elif isinstance(net, nn.Conv2d):
            if exists(net.bias):
                yield net.bias
            yield net.weight

        else:
            # Loop through children recursively
            has_children = False
            for m in net.children():
                has_children = True
                yield from params(m)
            # No children, yield parameters
            if not has_children:
                yield from net.state_dict().values() 
    
    with open(weights_path, "rb") as f:
        major, minor, _ = np.fromfile(f, dtype=np.int32, count=3)
        steps = np.fromfile(f, count=1, dtype=np.int64 if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000 else np.int32)
        offset = f.tell()

        # Load all weights
        for w in params(net):
            w.data.copy_(torch.from_numpy(np.fromfile(f, dtype=np.float32, count=w.numel())).view_as(w))

        assert (nP1 := count_parameters(net) * 4) == (nP2 := f.tell() - offset), f"{nP1} != {nP2}"


def load_from_ultralytics(net: Union[Yolov5, Yolov8, Yolov10, Yolov11]):
    from ultralytics import YOLO

    if isinstance(net, Yolov5):
        net2  = YOLO('yolov5{}u.pt'.format(net.v)).model.eval()
        l0,l1 = 10,24
    elif isinstance(net, Yolov8):
        net2  = YOLO('yolov8{}.pt'.format(net.v)).model.eval()
        l0,l1 = 10,22
    elif isinstance(net, Yolov10):
        net2  = YOLO('yolov10{}.pt'.format(net.v)).model.eval()
        l0,l1 = 11,23
    elif isinstance(net, Yolov11):
        net2  = YOLO('yolo11{}.pt'.format(net.v)).model.eval()
        l0,l1 = 11,23

    assert (nP1 := count_parameters(net)) == (nP2 := count_parameters(net2)), f'wrong number of parameters net {nP1} vs ultralytics {nP2}'
    copy_params(net.net, net2.model[0:l0])
    copy_params(net.fpn, net2.model[l0:l1])
    copy_params(net.head.cv2, net2.model[l1].cv2)
    copy_params(net.head.cv3, net2.model[l1].cv3)
    if hasattr(net.head, 'one2one_cv2'):
        copy_params(net.head.one2one_cv2, net2.model[l1].one2one_cv2)
        copy_params(net.head.one2one_cv3, net2.model[l1].one2one_cv3)
    for module in net.modules():
        if isinstance(module, C3):
            swap_convs(module.cv1, module.cv2)


def load_from_yolov6_official(net: Yolov6, weights_pt: str):
    def params(n):
        # Handle special modules (we've ordered submodules differently)
        if isinstance(n, RepConv):
            if n.bn is not None:
                yield from params(n.bn)
            yield from params(n.c1)
            yield from params(n.c2)
        elif isinstance(n, BottleRep):
            if isinstance(n.alpha, nn.Parameter):
                yield n.alpha
            yield from params(n.conv1)
            yield from params(n.conv2)
        # Loop through children recursively
        else:
            has_children = False
            for m in n.children():
                has_children = True
                yield from params(m)
            # No children, yield parameters
            if not has_children:
                yield from n.state_dict().values() 
    
    state = torch.load(weights_pt, map_location='cpu', weights_only=True)
    del state['detect.proj']
    del state['detect.proj_conv.weight']
    assert (nP1 := sum(p.numel() for p in params(net))) == (nP2 := sum(p.numel() for p in state.values())), f"{nP1} != {nP2}"

    for p1, (k, p2) in zip(params(net), state.items(), strict=True):
        assert p1.shape == p2.shape, f"bad shape: {k} {p2.shape} {p1.shape}"
        p1.data.copy_(p2.data)

    init_batchnorms(net)


def load_from_yolov7_official(net: Yolov7, weights_pt: str):
    def params1():
        for l in net.layers():
            for k, v in l.state_dict().items():
                if 'anchor' not in k:
                    yield v
    
    def params2():
        state2 = torch.load(weights_pt, map_location='cpu', weights_only=True)
        for k, v in state2.items():
            if 'anchor' not in k:
                yield v

    for p1, p2 in zip(params1(), params2(), strict=True):
        p1.data.copy_(p2.data)

    init_batchnorms(net)

    # Handle special case in SPPCSPC where Yolov6 and Yolov7 disagree on the order of the final torch.cat()
    for module in net.modules():
        if isinstance(module, SPPCSPC):
            layer  = module.cv7[0]
            weight = layer.weight.chunk(2, 1)
            layer.weight.data.copy_(torch.cat([weight[1], weight[0]], 1))


def get_model(model: str, variant: str = ''):
    match model:
        case 'yolov3' :     net = Yolov3(80, False).eval()
        case 'yolov3-spp':  net = Yolov3(80, True).eval()
        case 'yolov3-tiny': net = Yolov3Tiny(80).eval()
        case 'yolov4':      net = Yolov4(80).eval()
        case 'yolov4-tiny': net = Yolov4Tiny(80).eval()
        case 'yolov5':      net = Yolov5(variant, 80).eval()
        case 'yolov6':      net = Yolov6(variant, 80).eval()
        case 'yolov7':      net = Yolov7(80).eval()
        case 'yolov8':      net = Yolov8(variant, 80).eval()
        case 'yolov10':     net = Yolov10(variant, 80).eval()
        case 'yolov11':     net = Yolov11(variant, 80).eval()
    
    print(f"{model}{variant} has {count_parameters(net)} parameters")

    os.makedirs('../weights', exist_ok=True)
    has_obj = True
    
    if 'yolov3' in model or 'yolov4' in model :
        filepath = f'../weights/{model}.weights'
        download_if_not_exist(model, filepath)
        load_from_darknet(net, filepath)
    
    if model in ['yolov5', 'yolov8', 'yolov10', 'yolov11']:
        load_from_ultralytics(net)
        has_obj = False

    elif model == 'yolov6':
        load_from_yolov6_official(net, f"../weights/yolov6{variant}.pt")
        has_obj = False

    elif model == 'yolov7':
        load_from_yolov7_official(net, '../weights/yolov7.pt')

    return net, has_obj


@torch.inference_mode
def test(model: str, variant: str = ''):
    net, has_obj = get_model(model, variant)

    img      = torchvision.io.read_image('../images/dog.jpg')
    _, preds = nms(net(img[None] / 255.0), 0.3, 0.5, has_objectness=has_obj)
    boxes    = preds[:,:4]
    cls      = preds[:,-80:].argmax(-1)
    canvas   = torchvision.utils.draw_bounding_boxes(img, boxes, [COCO_NAMES[i] for i in cls])
    torchvision.io.write_png(canvas, f'dog_{model}{variant}_output.png')


def export(model: str, variant: str = '', slim=True):
    net, has_obj = get_model(model, variant)
    x = torch.randn(4, 3, 640, 640)
    _ = net(x) # warmup all the einops kernels

    print(bcolors.OKGREEN, f"Exporting {type(net).__name__} ...", bcolors.ENDC)
    torch.onnx.export(net, (x,), '/tmp/model.onnx',
                      input_names=['img'],
                      output_names=['preds'],
                      dynamic_axes={'img'   : {0: 'B', 2: 'H', 3: 'W'},
                                    'preds' : {0: 'B', 1: 'N'}})
    print(bcolors.OKGREEN, f"Exporting {type(net).__name__} ... Done", bcolors.ENDC)

    if slim:
        print(bcolors.OKGREEN, f"Slimming {type(net).__name__} ...", bcolors.ENDC)
        model_onnx = onnx.load('/tmp/model.onnx') 
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, '/tmp/model.onnx')
        print(bcolors.OKGREEN, f"Slimming {type(net).__name__} ... Done", bcolors.ENDC)

    print(bcolors.OKGREEN, "Checking with onnxruntime...", bcolors.ENDC)
    netOrt  = ort.InferenceSession('/tmp/model.onnx', providers=['CPUExecutionProvider'])
    x       = torch.randn(1, 3, 576, 768)
    preds1  = net(x) 
    preds2, = netOrt.run(None, {'img': x.numpy()})
    torch.testing.assert_close(preds1, torch.from_numpy(preds2)) #, atol=5e-5, rtol=1e-4)
    print(bcolors.OKGREEN, "Checking with onnxruntime... Done", bcolors.ENDC)


# def export_tflite(model: str, variant: str = ''):
#     import ai_edge_torch

#     net, has_obj = get_model(model, variant)
#     x = torch.randn(4, 3, 640, 640)
#     _ = net(x) # warmup all the einops kernels

#     print(bcolors.OKGREEN, f"Exporting {type(net).__name__} to TFLITE ...", bcolors.ENDC)
#     edge_model = ai_edge_torch.convert(net.eval(), (x,))
#     edge_model.export('/tmp/model.tflite')
#     print(bcolors.OKGREEN, f"Exporting {type(net).__name__} to TFLITE ... Done", bcolors.ENDC)

#     print(bcolors.OKGREEN, "Checking with ai_edge_torch ...", bcolors.ENDC)
#     x       = torch.randn(1, 3, 576, 768)
#     preds1  = net(x) 
#     preds2, = edge_model(x)
#     torch.testing.assert_close(preds1, preds2) #, atol=5e-5, rtol=1e-4)
#     print(bcolors.OKGREEN, "Checking with ai_edge_torch... Done", bcolors.ENDC)

test('yolov3')
test('yolov3-spp')
test('yolov3-tiny')
test('yolov4')
test('yolov4-tiny')
test('yolov5', 'n')
test('yolov5', 's')
test('yolov5', 'm')
test('yolov5', 'l')
test('yolov5', 'x')
test('yolov6', 'n')
test('yolov6', 's')
test('yolov6', 'm')
test('yolov7')
test('yolov8', 'n')
test('yolov8', 's')
test('yolov8', 'm')
test('yolov8', 'l')
test('yolov8', 'x')
test('yolov10', 'n')
test('yolov10', 's')
test('yolov10', 'm')
test('yolov10', 'b')
test('yolov10', 'l')
test('yolov10', 'x')
test('yolov11', 'n')
test('yolov11', 's')
test('yolov11', 'm')
test('yolov11', 'l')
test('yolov11', 'x')

# export('yolov3')
# export('yolov3-spp')
# export('yolov3-tiny')
# export('yolov4')
# export('yolov4-tiny')
# export('yolov5', 'n')
# export('yolov5', 's')
# export('yolov5', 'm')
# export('yolov5', 'l')
# export('yolov5', 'x')
# export('yolov6', 'n')
# export('yolov6', 's')
# export('yolov6', 'm')
# export('yolov7')
# export('yolov8', 'n')
# export('yolov8', 's')
# export('yolov8', 'm')
# export('yolov8', 'l')
# export('yolov8', 'x')
# export('yolov10', 'n')
# export('yolov10', 's')
# export('yolov10', 'm')
# export('yolov10', 'b')
# export('yolov10', 'l')
# export('yolov10', 'x')