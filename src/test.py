import  time
import  torch
import  torchvision
import  torchvision.transforms.functional as vF
import  matplotlib.pyplot as plt
from    models import *

def test(type: str, size: str = ''):
    match type:
        case 'yolov3' :     net = Yolov3(80, False).eval()
        case 'yolov3-spp':  net = Yolov3(80, True).eval()
        case 'yolov3-tiny': net = Yolov3Tiny(80).eval()
        case 'yolov4':      net = Yolov4(80).eval()
        case 'yolov4-tiny': net = Yolov4Tiny(80).eval()
        case 'yolov5':      net = Yolov5(size, 80).eval()
        case 'yolov8':      net = Yolov8(size, 80).eval()
        case 'yolov7':      net = Yolov7(80).eval()
        case 'yolov10':     net = Yolov10(size, 80).eval()
    
    print("{}{} has {} parameters".format(type, size, count_parameters(net)))

    has_obj = True

    if type in ['yolov5', 'yolov8', 'yolov10']:
        load_from_ultralytics(net)
        has_obj = False
    
    elif 'yolov3' in type or 'yolov4' in type :
        load_darknet(net, '../weights/{}.weights'.format(type))
    
    elif type == 'yolov7':
        load_yolov7(net, '../weights/yolov7.pt')

    img = torchvision.io.read_image('../images/dog.jpg')
    _, preds = nms(net(img[None] / 255.0), 0.3, 0.5, has_objectness=has_obj)
    off   = 5 if has_obj else 4
    boxes = preds[:,:4]
    cls   = preds[:,off:].argmax(-1)
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
test('yolov7')
test('yolov10', 'n')
test('yolov10', 's')
test('yolov10', 'm')
test('yolov10', 'b')
test('yolov10', 'l')
test('yolov10', 'x')