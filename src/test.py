import  os
import  torch
import  torchvision
from    models import *

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


def load_from_darknet(net: Union[Yolov3, Yolov3Tiny, Yolov4, Yolov4Tiny], weights_path: str):
    with open(weights_path, "rb") as f:
        major, minor, _ = np.fromfile(f, dtype=np.int32, count=3)
        steps = np.fromfile(f, count=1, dtype=np.int64 if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000 else np.int32)
        offset = f.tell()

        # Get all weights
        weights = []
        for block in net.layers():
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    weights.append([m.bias, m.weight] if exists(m.bias) else [m.weight])
                if isinstance(m, nn.BatchNorm2d):
                    conv_weights = weights.pop()
                    weights.append([m.bias, m.weight, m.running_mean, m.running_var])
                    weights.append(conv_weights)

        # Load all weights
        for w in weights:
            for wi in w:
                wi.data.copy_(torch.from_numpy(np.fromfile(f, dtype=np.float32, count=wi.numel())).view_as(wi))

        assert (nP1 := count_parameters(net) * 4) == (nP2 := f.tell() - offset), f"{nP1} != {nP2}"


def load_from_ultralytics(net: Union[Yolov5, Yolov8, Yolov10]):
    from ultralytics import YOLO

    if isinstance(net, Yolov5):
        net2 = YOLO('yolov5{}u.pt'.format(net.v)).model.eval()
        assert (nP1 := count_parameters(net)) == (nP2 := count_parameters(net2)), f'wrong number of parameters net {nP1} vs ultralytics {nP2}'
        copy_params(net.net, net2.model[0:10])
        copy_params(net.fpn, net2.model[10:24])
        copy_params(net.head.cv2, net2.model[24].cv2)
        copy_params(net.head.cv3, net2.model[24].cv3)

    elif isinstance(net, Yolov8):
        net2 = YOLO('yolov8{}.pt'.format(net.v)).model.eval()
        assert (nP1 := count_parameters(net)) == (nP2 := count_parameters(net2)), 'wrong number of parameters net {} vs ultralytics {}'.format(nP1, nP2)
        copy_params(net.net, net2.model[0:10])
        copy_params(net.fpn, net2.model[10:22])
        copy_params(net.head.cv2, net2.model[22].cv2)
        copy_params(net.head.cv3, net2.model[22].cv3)
    
    elif isinstance(net, Yolov10):
        net2 = YOLO('yolov10{}.pt'.format(net.v)).model.eval()
        assert (nP1 := count_parameters(net)) == (nP2 := count_parameters(net2)), 'wrong number of parameters net {} vs ultralytics {}'.format(nP1, nP2)
        copy_params(net.net, net2.model[0:11])
        copy_params(net.fpn, net2.model[11:23])
        copy_params(net.head.cv2, net2.model[23].cv2)
        copy_params(net.head.cv3, net2.model[23].cv3)
        copy_params(net.head.one2one_cv2, net2.model[23].one2one_cv2)
        copy_params(net.head.one2one_cv3, net2.model[23].one2one_cv3)


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


def test(type: str, size: str = ''):
    os.makedirs('../weights', exist_ok=True)

    match type:
        case 'yolov3' :     net = Yolov3(80, False).eval()
        case 'yolov3-spp':  net = Yolov3(80, True).eval()
        case 'yolov3-tiny': net = Yolov3Tiny(80).eval()
        case 'yolov4':      net = Yolov4(80).eval()
        case 'yolov4-tiny': net = Yolov4Tiny(80).eval()
        case 'yolov5':      net = Yolov5(size, 80).eval()
        case 'yolov6':      net = Yolov6(size, 80).eval()
        case 'yolov7':      net = Yolov7(80).eval()
        case 'yolov8':      net = Yolov8(size, 80).eval()
        case 'yolov10':     net = Yolov10(size, 80).eval()
    
    print("{}{} has {} parameters".format(type, size, count_parameters(net)))

    has_obj = True

    if type in ['yolov5', 'yolov8', 'yolov10']:
        load_from_ultralytics(net)
        has_obj = False
    
    elif 'yolov3' in type or 'yolov4' in type :
        filepath = '../weights/{}.weights'.format(type)
        download_if_not_exist(type, filepath)
        load_from_darknet(net, filepath)
    
    elif type == 'yolov6':
        load_from_yolov6_official(net, "../weights/yolov6{}.pt".format(size))
        has_obj = False

    elif type == 'yolov7':
        load_from_yolov7_official(net, '../weights/yolov7.pt')

    img = torchvision.io.read_image('../images/dog.jpg')
    _, preds = nms(net(img[None] / 255.0), 0.3, 0.5, has_objectness=has_obj)
    boxes = preds[:,:4]
    cls   = preds[:,-80:].argmax(-1)
    canvas = torchvision.utils.draw_bounding_boxes(img, boxes, [COCO_NAMES[i] for i in cls])
    torchvision.io.write_png(canvas, 'dog_{}{}_output.png'.format(type, size))

test('yolov5', 'n')
test('yolov5', 's')
test('yolov5', 'm')
test('yolov5', 'l')
test('yolov5', 'x')
test('yolov8', 'n')
test('yolov8', 's')
test('yolov8', 'm')
test('yolov8', 'l')
test('yolov8', 'x')
test('yolov3')
test('yolov3-spp')
test('yolov3-tiny')
test('yolov4')
test('yolov4-tiny')
test('yolov6', 'n')
test('yolov6', 's')
test('yolov6', 'm')
test('yolov7')
test('yolov10', 'n')
test('yolov10', 's')
test('yolov10', 'm')
test('yolov10', 'b')
test('yolov10', 'l')
test('yolov10', 'x')