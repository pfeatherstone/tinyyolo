import math
from   typing import Union, List
from   copy import deepcopy
from   functools import partial
import numpy as np
import numba as nb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from   einops import rearrange, repeat

COCO_NAMES = [
    'person',           'bicycle',      'car',          'motorbike',    'aeroplane',    
    'bus',              'train',        'truck',        'boat',         'traffic light',
    'fire hydrant',     'stop sign',    'parking meter','bench',        'bird',
    'cat',              'dog',          'horse',        'sheep',        'cow',
    'elephant',         'bear',         'zebra',        'giraffe',      'backpack',
    'umbrella',         'handbag',      'tie',          'suitcase',     'frisbee',
    'skis',             'snowboard',    'sports ball',  'kite',         'baseball bat',
    'baseball glove',   'skateboard',   'surfboard',    'tennis racket','bottle',
    'wine glass',       'cup',          'fork',         'knife',        'spoon',
    'bowl',             'banana',       'apple',        'sandwich',     'orange',
    'broccoli',         'carrot',       'hot dog',      'pizza',        'donut',
    'cake',             'chair',        'sofa',         'pottedplant',  'bed',
    'diningtable',      'toilet',       'tvmonitor',    'laptop',       'mouse',
    'remote',           'keyboard',     'cell phone',   'microwave',    'oven',
    'toaster',          'sink',         'refrigerator', 'book',         'clock',
    'vase',             'scissors',     'teddy bear',   'hair drier',   'toothbrush'
]

ANCHORS_V3      = [[(10,13), (16,30), (33,23)], [(30,61), (62,45), (59,119)], [(116,90), (156,198), (373,326)]]
ANCHORS_V3_TINY = [[(10,14), (23,27), (37,58)], [(81,82), (135,169), (344,319)]]
ANCHORS_V4      = [[(12,16), (19,36), (40,28)], [(36,75), (76,55), (72, 146)], [(142,110), (192,243), (459,401)]]
ANCHORS_V7      = ANCHORS_V4

actV3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
actV4 = nn.Mish(inplace=True)

def get_variant_multiplesV5(variant):
    return {'n':(0.33, 0.25, 2.0), 
            's':(0.33, 0.50, 2.0), 
            'm':(0.67, 0.75, 2.0), 
            'l':(1.00, 1.00, 2.0), 
            'x':(1.33, 1.25, 2.0) }.get(variant, None)

def get_variant_multiplesV8(variant):
    return {'n':(0.33, 0.25, 2.0), 
            's':(0.33, 0.50, 2.0), 
            'm':(0.67, 0.75, 1.5), 
            'l':(1.00, 1.00, 1.0), 
            'x':(1.00, 1.25, 1.0) }.get(variant, None)

def batchnorms(n: nn.Module):
    for m in n.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            yield m
            
def copy_params(n1: nn.Module, n2: nn.Module):
    for p1, p2 in zip(n1.parameters(), n2.parameters(), strict=True):
        p1.data.copy_(p2.data)
    
    # running statistics aren't included in nn.Module.parameters()
    for m1, m2 in zip(batchnorms(n1), batchnorms(n2), strict=True):
        m1.running_mean.data.copy_(m2.running_mean.data)
        m1.running_var.data.copy_(m2.running_var.data)
   
def count_parameters(net: torch.nn.Module, include_stats=True):
    return sum(p.numel() for p in net.parameters()) + (sum(m.running_mean.numel() + m.running_var.numel() for m in batchnorms(net)) if include_stats else 0)
        
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Residual(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self, x):
        return x + self.f(x)

def Repeat(module, N):
    return nn.Sequential(*[deepcopy(module) for _ in range(N)])

def Conv(c1, c2, k=1, s=1, p=None, act=nn.SiLU(True)):
    return nn.Sequential(nn.Conv2d(c1, c2, k, s, default(p,k//2), bias=False),
                         nn.BatchNorm2d(c2),
                         act)

def Con5(c1, c2=None, spp=False, act=actV3):
    c2   = c2 if exists(c2) else c1
    conv = partial(Conv, act=act)
    return nn.Sequential(conv(c1, c2//2, 1),
                         conv(c2//2, c2, 3),
                         conv(c2, c2//2, 1),
                         Spp(c2//2, act=actV3) if spp else nn.Identity(),
                         conv(c2//2, c2, 3),
                         conv(c2, c2//2, 1)) 

def Bottleneck(c1, c2=None, k=(3, 3), shortcut=True, e=0.5, act=nn.SiLU(True)):
    c2  = default(c2, c1)
    c_  = int(c2 * e)
    net = nn.Sequential(Conv(c1, c_, k[0], act=act), Conv(c_, c2, k[1], act=act))
    return Residual(net) if shortcut else net

def MaxPool(stride):
    return nn.Sequential(nn.ZeroPad2d((0,1,0,1)), nn.MaxPool2d(kernel_size=2, stride=stride))

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)f
        self.m   = Repeat(Bottleneck(c_, shortcut=shortcut, k=(1, 3), e=1.0), n)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c   = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m   = nn.ModuleList(Bottleneck(self.c, k=(3, 3), e=1.0, shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Spp(nn.Module):
    def __init__(self, inc, act):
        super().__init__()
        self.conv = Conv(inc*4, inc, 1, act=act)
    def forward(self, x):
        a = torch.max_pool2d(x, 5, 1, 2, 1, True)
        b = torch.max_pool2d(x, 9, 1, 4, 1, True)
        c = torch.max_pool2d(x, 13, 1, 6, 1, True)
        d = torch.cat((c, b, a, x), dim=1)
        x = self.conv(d)
        return x
    
class SPPF(nn.Module):
    def __init__(self, c1, c2):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_          = c1 // 2  # hidden channels
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
    def forward(self, x):
        x  = self.cv1(x)
        y1 = torch.max_pool2d(x, 5, 1, 2)
        y2 = torch.max_pool2d(y1, 5, 1, 2)
        y3 = torch.max_pool2d(y2, 5, 1, 2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))

class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, e=0.5, k=(5, 9, 13)):
        super().__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        conv = partial(Conv, act=actV3)
        res  = partial(Bottleneck, act=actV3, k=(1, 3))
        self.conv   = conv(3, 32, 3)
        self.block1 = nn.Sequential(conv( 32,   64, 3, 2), res(64))
        self.block2 = nn.Sequential(conv( 64,  128, 3, 2), Repeat(res(128), 2))
        self.block3 = nn.Sequential(conv(128,  256, 3, 2), Repeat(res(256), 8))
        self.block4 = nn.Sequential(conv(256,  512, 3, 2), Repeat(res(512), 8))
        self.block5 = nn.Sequential(conv(512, 1024, 3, 2), Repeat(res(1024),4))
    def forward(self, x):
        x8  = self.block3(self.block2(self.block1(self.conv(x))))
        x16 = self.block4(x8)
        x32 = self.block5(x16)
        return x8, x16, x32

class BackboneV3Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        conv = partial(Conv, act=actV3)
        self.b1 = nn.Sequential(conv(  3,  16, 3), MaxPool(2),
                                conv( 16,  32, 3), MaxPool(2),
                                conv( 32,  64, 3), MaxPool(2),
                                conv( 64, 128, 3), MaxPool(2),
                                conv(128, 256, 3))
        self.b2 = nn.Sequential(MaxPool(2), conv(256, 512, 3), MaxPool(1), conv(512, 1024, 3))
    def forward(self, x):
        x16 = self.b1(x)
        x32 = self.b2(x16)
        return x16, x32

class CspBlock(nn.Module):
    def __init__(self, c1, c2, f=1, e=1, act=actV3, n=1):
        super().__init__()
        conv = partial(Conv, act=act)
        c_   = int(c2*f)
        self.d  = conv(c1, c_, 3, s=2)
        self.c1 = conv(c_, c2, 1)
        self.c2 = conv(c_, c2, 1)
        self.m  = Repeat(Bottleneck(c2, e=e, k=(1,3), act=act), n)
        self.c3 = conv(c2, c2, 1)
        self.c4 = conv(2*c2, c_, 1)
    def forward(self, x):
        x = self.d(x)
        a = self.c1(x)
        b = self.c3(self.m(self.c2(x)))
        x = torch.cat([b, a], 1)
        x = self.c4(x)
        return x

class BackboneV4(nn.Module):
    def __init__(self, act):
        super().__init__()
        conv = partial(Conv, act=act)
        csp  = partial(CspBlock, act=act)
        self.stem = conv(3, 32, 3)
        self.b1   = csp( 32,  64, f=1, e=0.5, n=1)
        self.b2   = csp( 64,  64, f=2, e=1,   n=2)
        self.b3   = csp(128, 128, f=2, e=1,   n=8)
        self.b4   = csp(256, 256, f=2, e=1,   n=8)
        self.b5   = csp(512, 512, f=2, e=1,   n=4)
    def forward(self, x):
        p8  = self.b3(self.b2(self.b1(self.stem(x))))
        p16 = self.b4(p8)
        p32 = self.b5(p16)
        return p8, p16, p32

class BackboneV4TinyBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        conv = partial(Conv, act=actV3)
        self.c0 = conv(c, c, 3, 1)
        self.c1 = conv(c, c, 3, 1)
        self.c2 = conv(c*2, c*2, 1, 1)
    def forward(self, x):
        a   = self.c0(x.chunk(2,dim=1)[1])
        b   = self.c1(a)
        c   = self.c2(torch.cat([b,a], 1))
        d   = torch.cat([x,c], 1) 
        return d

class BackboneV4Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        conv = partial(Conv, act=actV3)
        self.b1 = nn.Sequential(conv(3, 32, 3, 2), conv(32, 64, 3, 2), conv(64, 64, 3, 1))
        self.c1 = BackboneV4TinyBlock(32)
        self.b2 = nn.Sequential(MaxPool(2), conv(128, 128, 3, 1))
        self.c2 = BackboneV4TinyBlock(64)
        self.b3 = nn.Sequential(MaxPool(2), conv(256, 256, 3, 1))
        self.c3 = BackboneV4TinyBlock(128)
        self.b4 = nn.Sequential(MaxPool(2), conv(512, 512, 3, 1))
    def forward(self, x):
        p8  = self.b2(self.c1(self.b1(x)))
        p16 = self.b3(self.c2(p8))
        p32 = self.b4(self.c3(p16))
        return p16, p32

class ElanBlock(nn.Module):
    def __init__(self, cin, mr, br, nb, nc, cout=None):
        super().__init__()
        cmid = int(mr*cin)
        cblk = int(br*cin)
        cfin = nb*cblk + 2*cmid
        cout = default(cout, 2*cin)
        self.c1 = Conv(cin, cmid, 1)
        self.cs = nn.ModuleList([Conv(cin, cmid, 1)] +
                                [nn.Sequential(*[Conv(cmid if j==0 and i==0 else cblk, cblk, 3) for j in range(nc)]) for i in range(nb)])
        self.c3 = Conv(cfin, cout, 1)
    def forward(self, x):
        xs = [self.c1(x)] + [x := c(x) for c in self.cs]
        x  = torch.cat(xs[::-1], 1)
        return self.c3(x)

class MaxPoolAndStrideConv(nn.Module):
    def __init__(self, cin, cout=None, use=False):
        super().__init__()
        cout    = cout if exists(cout) else cin
        cmid    = cin if use else cout // 2
        self.b1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), Conv(cin, cout//2, 1))
        self.b2 = nn.Sequential(Conv(cin, cmid, 1), Conv(cmid, cout//2, 3, s=2))
    def forward(self, x):
        return torch.cat([self.b2(x), self.b1(x)], 1)
    
class RepConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = Conv(c1, c2, 3, act=nn.Identity())
        self.c2 = Conv(c1, c2, 1, act=nn.Identity())
        self.bn = nn.BatchNorm2d(c1) if c1==c2 else None
    def forward(self, x):
        id_out = self.bn(x) if exists(self.bn) else 0
        return F.silu(self.c1(x) + self.c2(x) + id_out, inplace=True)
    
class Scale(nn.Module):
    def __init__(self, c, add: bool):
        super().__init__()
        self.add = add
        self.g   = nn.Parameter(torch.randn(1, c, 1, 1))
    def forward(self, x):
        return (x + self.g) if self.add else (x * self.g)
    
class BackboneV7(nn.Module):
    def __init__(self):
        super().__init__()
        self.b0 = nn.Sequential(Conv(3, 32, 3), Conv(32, 64, 3, s=2), Conv(64, 64, 3))
        self.b1 = nn.Sequential(Conv(64, 128, 3, s=2), ElanBlock(128, mr=0.5, br=0.5, nb=2, nc=2))
        self.b2 = nn.Sequential(MaxPoolAndStrideConv(256), ElanBlock(256, mr=0.5, br=0.5, nb=2, nc=2))
        self.b3 = nn.Sequential(MaxPoolAndStrideConv(512), ElanBlock(512, mr=0.5, br=0.5, nb=2, nc=2))
        self.b4 = nn.Sequential(MaxPoolAndStrideConv(1024), ElanBlock(1024, mr=0.25, br=0.25, nb=2, nc=2, cout=1024))
    def forward(self, x):
        p8  = self.b2(self.b1(self.b0(x)))
        p16 = self.b3(p8)
        p32 = self.b4(p16)
        return p8, p16, p32

class BackboneV5(nn.Module):
    def __init__(self, w, r, d):
        super().__init__()
        self.b0 = Conv(c1=3, c2=int(64*w), k=6, s=2, p=2)
        self.b1 = Conv(int(64*w), int(128*w), k=3, s=2)
        self.b2 = C3(c1=int(128*w), c2=int(128*w), n=round(3*d))
        self.b3 = Conv(int(128*w), int(256*w), k=3, s=2)
        self.b4 = C3(c1=int(256*w), c2=int(256*w), n=round(6*d))
        self.b5 = Conv(int(256*w), int(512*w), k=3, s=2)
        self.b6 = C3(c1=int(512*w), c2=int(512*w), n=round(9*d))
        self.b7 = Conv(int(512*w), int(512*w*r), k=3, s=2)
        self.b8 = C3(c1=int(512*w*r), c2=int(512*w*r), n=round(3*d))
        self.b9 = SPPF(int(512*w*r), int(512*w*r))

    def forward(self, x):
        x4 = self.b4(self.b3(self.b2(self.b1(self.b0(x))))) # 4 P3/8
        x6 = self.b6(self.b5(x4))                           # 6 P4/16
        x9 = self.b9(self.b8(self.b7(x6)))                  # 9 P5/32
        return x4, x6, x9

class BackboneV8(nn.Module):
    def __init__(self, w, r, d):
        super().__init__()
        self.b0 = Conv(c1=3, c2= int(64*w), k=3, s=2)
        self.b1 = Conv(int(64*w), int(128*w), k=3, s=2)
        self.b2 = C2f(c1=int(128*w), c2=int(128*w), n=round(3*d), shortcut=True)
        self.b3 = Conv(int(128*w), int(256*w), k=3, s=2)
        self.b4 = C2f(c1=int(256*w), c2=int(256*w), n=round(6*d), shortcut=True)
        self.b5 = Conv(int(256*w), int(512*w), k=3, s=2)
        self.b6 = C2f(c1=int(512*w), c2=int(512*w), n=round(6*d), shortcut=True)
        self.b7 = Conv(int(512*w), int(512*w*r), k=3, s=2)
        self.b8 = C2f(c1=int(512*w*r), c2=int(512*w*r), n=round(3*d), shortcut=True)
        self.b9 = SPPF(int(512*w*r), int(512*w*r))

    def forward(self, x):
        x4 = self.b4(self.b3(self.b2(self.b1(self.b0(x))))) # 4 P3/8
        x6 = self.b6(self.b5(x4))                           # 6 P4/16
        x9 = self.b9(self.b8(self.b7(x6)))                  # 9 P5/32
        return x4, x6, x9

class HeadV3(nn.Module):
    def __init__(self, spp):
        super().__init__() 
        self.b1 = Con5(1024, 1024, spp, act=actV3)
        self.c1 = Conv(512, 256, 1,  act=actV3)
        self.b2 = Con5(512+256, 512, act=actV3)
        self.c2 = Conv(256, 128, 1,  act=actV3)
        self.b3 = Con5(256+128, 256, act=actV3)
    def forward(self, x8, x16, x32):
        p32 = self.b1(x32)
        p16 = self.b2(torch.cat([F.interpolate(self.c1(p32), scale_factor=2), x16], 1))
        p8  = self.b3(torch.cat([F.interpolate(self.c2(p16), scale_factor=2), x8], 1))
        return p8, p16, p32
    
class HeadV3Tiny(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = Conv(c, 256, 1, act=actV3)
        self.c2 = Conv(256,  128, 1, act=actV3)
    def forward(self, x16, x32):
        p32 = self.b1(x32)
        p16 = torch.cat([F.interpolate(self.c2(p32), scale_factor=2), x16], 1)
        return p16, p32

class HeadV4(nn.Module):
    def __init__(self, act):
        super().__init__()
        conv, con5 = partial(Conv, act=act), partial(Con5, act=act)
        self.b1 = con5(1024, spp=True)
        self.c1 = nn.ModuleList([conv(512, 256, 1) for _ in range(2)])
        self.b2 = con5(512, spp=False)
        self.c2 = nn.ModuleList([conv(256, 128, 1) for _ in range(2)])
        self.b3 = con5(256, spp=False)
        self.c3 = conv(128, 256, 3, 2)
        self.b4 = con5(512, spp=False)
        self.c4 = conv(256, 512, 3, 2)
        self.b5 = con5(1024, spp=False)
    
    def forward(self, x8, x16, x32):
        p32 = self.b1(x32)
        p16 = self.b2(torch.cat([self.c1[1](x16), F.interpolate(self.c1[0](p32), scale_factor=2)], 1))
        n8  = self.b3(torch.cat([self.c2[1](x8),  F.interpolate(self.c2[0](p16), scale_factor=2)], 1))
        n16 = self.b4(torch.cat([self.c3(n8),  p16], 1))
        n32 = self.b5(torch.cat([self.c4(n16), p32], 1))
        return n8, n16, n32

class HeadV7(nn.Module):
    def __init__(self):
        super().__init__() # ordering of layers is set to ease loading of original Yolov7 weigths
        self.c13 = SPPCSPC(1024, 512)
        self.c14 = Conv(512, 256, 1)
        self.c12 = Conv(1024, 256, 1)
        self.c17 = ElanBlock(512, mr=0.5, br=0.25, nb=4, nc=1, cout=256)
        self.c18 = Conv(256, 128, 1)
        self.c11 = Conv(512, 128, 1)
        self.cxx = ElanBlock(256, mr=0.5, br=0.25, nb=4, nc=1, cout=128)
        self.c21 = MaxPoolAndStrideConv(128, 256)
        self.c23 = ElanBlock(512, mr=0.5, br=0.25, nb=4, nc=1, cout=256)
        self.c24 = MaxPoolAndStrideConv(256, 512)
        self.c26 = ElanBlock(1024, mr=0.5, br=0.25, nb=4, nc=1, cout=512)

    def forward(self, p8, p16, p32):
        x13 = self.c13(p32)
        x16 = torch.cat((self.c12(p16), F.interpolate(self.c14(x13), scale_factor=2)), 1)
        x17 = self.c17(x16)
        x20 = torch.cat((self.c11(p8), F.interpolate(self.c18(x17), scale_factor=2)), 1)
        xxx = self.cxx(x20)
        x23 = self.c23(torch.cat((self.c21(xxx), x17), 1))
        x26 = self.c26(torch.cat((self.c24(x23), x13), 1))
        return xxx, x23, x26

class HeadV5(nn.Module):
    def __init__(self, w, r, d):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.n0 = Conv(c1=int(512*w*r), c2=int(512*w))
        self.n1 = C3(c1=int(512*w*2), c2=int(512*w), n=round(3*d), shortcut=False)
        self.n2 = Conv(c1=int(512*w), c2=int(256*w))
        self.n3 = C3(c1=int(256*w*2), c2=int(256*w), n=round(3*d), shortcut=False)
        self.n4 = Conv(c1=int(256*w), c2=int(256*w), k=3, s=2)
        self.n5 = C3(c1=int(256*w*2), c2=int(512*w), n=round(3*d), shortcut=False)
        self.n6 = Conv(c1=int(512* w), c2=int(512*w), k=3, s=2)
        self.n7 = C3(c1=int(512*w)*2, c2=int(512*w*2), n=round(3*d), shortcut=False)

    def forward(self, x4, x6, x9):
        x10 = self.n0(x9)
        x13 = self.n1(torch.cat([self.up(x10),x6], 1))
        x14 = self.n2(x13)                                      
        x17 = self.n3(torch.cat([self.up(x14),x4], 1))  # 17 (P3/8-small)
        x20 = self.n5(torch.cat([self.n4(x17),x14], 1))
        x23 = self.n7(torch.cat([self.n6(x20), x10], 1))
        return [x17, x20, x23]

class HeadV8(nn.Module):
    def __init__(self, w, r, d):  #width_multiple, ratio_multiple, depth_multiple
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.n1 = C2f(c1=int(512*w*(1+r)), c2=int(512*w), n=round(3*d))
        self.n2 = C2f(c1=int(768*w), c2=int(256*w), n=round(3*d))
        self.n3 = Conv(c1=int(256*w), c2=int(256*w), k=3, s=2)
        self.n4 = C2f(c1=int(768*w), c2=int(512*w), n=round(3*d))
        self.n5 = Conv(c1=int(512* w), c2=int(512 * w), k=3, s=2)
        self.n6 = C2f(c1=int(512*w*(1+r)), c2=int(512*w*r), n=round(3*d))

    def forward(self, x4, x6, x9):
        x12 = self.n1(torch.cat([self.up(x9),x6], 1))   # 12
        x15 = self.n2(torch.cat([self.up(x12),x4], 1))  # 15 (P3/8-small)
        x18 = self.n4(torch.cat([self.n3(x15),x12], 1)) # 18 (P4/16-medium)
        x21 = self.n6(torch.cat([self.n5(x18), x9], 1)) # 21 (P5/32-large)
        return [x15, x18, x21]

def make_anchors(feats, strides, grid_cell_offset=0.5):
    sxy             = []
    strides_tensor  = []
    for x, stride in zip(feats, strides):
        h, w    = x.shape[2:]
        sx      = torch.arange(end=w, device=x.device) + grid_cell_offset  # shift x
        sy      = torch.arange(end=h, device=x.device) + grid_cell_offset  # shift y
        sy, sx  = torch.meshgrid(sy, sx, indexing='ij')
        sxy.append(torch.stack([sx,sy],0).flatten(-2))
        strides_tensor.append(torch.full((h*w,), fill_value=stride, device=x.device))
    return torch.cat(sxy, 1), torch.cat(strides_tensor)

class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc         = nc                        # number of classes
        self.reg_max    = 16                        # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no         = nc + self.reg_max * 4     # number of outputs per anchor
        self.strides    = [8, 16, 32]               # strides computed during build
        c2, c3          = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2        = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3        = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.r          = nn.Parameter(torch.arange(self.reg_max).float(), requires_grad=False)

    def forward(self, x, labels=None):
        for i in range(len(x)):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if exists(labels):
            return x
        
        sxy, strides = make_anchors(x, self.strides)
        x            = torch.cat([xi.flatten(2) for xi in x], 2)
        dist, cls    = x.split((4 * self.reg_max, self.nc), 1)
        dist         = rearrange(dist, 'b (k r) a -> b k r a', k=4).softmax(2)
        dist         = torch.einsum('bkra, r -> bka', dist, self.r)
        lt, rb       = dist.chunk(2, dim=1)
        x1y1         = sxy - lt
        x2y2         = sxy + rb
        box          = torch.cat([x1y1,x2y2],1) * strides
        preds        = torch.cat((box, cls.sigmoid()), 1)
        return preds.transpose(1,2)

@nb.njit
def build_targets_v3(B: int, H: int, W: int, C: int, stride: int, anchors: np.ndarray, targets: np.ndarray):
    def wh_iou(wh1, wh2):
        area1   = wh1[0] * wh1[1]
        area2   = wh2[:,0] * wh2[:,1]
        inter   = np.minimum(wh1, wh2)
        inter   = inter[:,0]*inter[:,1]
        union   = area1+area2-inter
        return inter/union

    target_tensor = np.zeros((B, anchors.shape[0], H, W, 6+C), dtype=np.float32)
    for b, tgt in enumerate(targets):
        for tgti in tgt:
            if tgti[4] > -1:
                box, c  = tgti[:4], int(tgti[4])
                wh      = box[2:]-box[:2]  
                x       = int(math.floor(0.5 * (box[0] + box[2]) / stride))
                y       = int(math.floor(0.5 * (box[1] + box[3]) / stride))
                ious    = wh_iou(wh, anchors)
                scores  = ious / np.max(ious)
                for a, (s1,s2) in enumerate(zip(ious, scores)):
                    if s1 > target_tensor[b, a, y, x, -1]:
                        target_tensor[b, a, y, x, :4]   = box
                        target_tensor[b, a, y, x, 4]    = s2
                        target_tensor[b, a, y, x, 5:]   = 0 #overwrite last cls index (and last iou value)
                        target_tensor[b, a, y, x, 5+c]  = 1
                        target_tensor[b, a, y, x, -1]   = s1

    return target_tensor[...,:-1]

class DetectV3(nn.Module):
    def __init__(self, nclasses, strides, anchors, scales, is_v7=False):
        super().__init__()
        self.register_buffer('anchors', torch.tensor(anchors).float())
        self.strides    = strides
        self.C          = nclasses
        self.scales     = scales
        self.v7         = is_v7

    @torch.no_grad()
    def make_grid(self, x):
        h, w    = x.shape[2], x.shape[3]
        sx      = torch.arange(end=w, device=x.device)
        sy      = torch.arange(end=h, device=x.device)
        sy, sx  = torch.meshgrid(sy, sx, indexing='ij')
        sxy     = torch.stack([sx,sy],-1)
        return sxy

    def to_xy(self, xy, scale, S):
        sxy = self.make_grid(xy)
        match self.v7:
            case True:  return S * (xy.sigmoid() * 2 - 0.5 + sxy)
            case False: return S * ((xy.sigmoid() - 0.5) * scale + 0.5 + sxy)

    def to_wh(self, wh, A):
        match self.v7:
            case True:  return A.view(1,-1,1,1,2) * (wh.sigmoid() ** 2) * 4
            case False: return A.view(1,-1,1,1,2) * wh.exp()
    
    def forward(self, *xs, targets=None):
        preds       = []
        loss_iou    = 0
        loss_cls    = 0
        loss_obj    = 0
        loss_noobj  = 0

        for x, stride, anchor, scale in zip(xs, self.strides, self.anchors, self.scales):
            dets            = rearrange(x, 'b (a f) h w -> b a h w f', f=self.C+5)
            xy, wh, l, cls  = dets.split((2,2,1,self.C), -1)
            xy              = self.to_xy(xy, scale, stride)
            wh              = self.to_wh(wh, anchor)
            box             = torch.cat([xy-wh/2, xy+wh/2], -1)
            pred            = torch.cat([box, l.sigmoid(), cls.sigmoid()], -1)
            pred            = rearrange(pred, 'b a h w f -> b (a h w) f')
            preds.append(pred)

            if exists(targets):
                tgt     = torch.from_numpy(build_targets_v3(x.shape[0], x.shape[2], x.shape[3], self.C, stride, anchor.cpu().numpy(), targets.cpu().numpy())).to(x.device)
                mask    = tgt[...,4] > 0
                tgt     = tgt[mask]
                pnoobj  = l[~mask]

                loss_iou    += torchvision.ops.complete_box_iou_loss(box[mask], tgt[:,:4], reduction='mean')
                loss_cls    += F.binary_cross_entropy_with_logits(cls[mask], tgt[:,5:])
                loss_obj    += F.binary_cross_entropy_with_logits(l[mask], tgt[:,4:5])
                loss_noobj  += F.binary_cross_entropy_with_logits(pnoobj, torch.zeros_like(pnoobj))
                
        preds  = torch.cat(preds, 1)
        losses = {'iou': loss_iou, 'cls': loss_cls, 'obj': loss_obj, 'noobj': loss_noobj} if exists(targets) else None
        return (preds, losses) if exists(losses) else preds
    
class Yolov3(nn.Module):
    def __init__(self, nclasses, spp):
        super().__init__()
        self.back = Darknet53()
        self.head = HeadV3(spp)
        self.neck = nn.ModuleList([nn.Sequential(Conv(c1, c2, 3, act=actV3), nn.Conv2d(c2, (nclasses+5)*3, 1)) for c1, c2 in [(128, 256), (256, 512), (512, 1024)]])
        self.yolo = DetectV3(nclasses, [8,16,32], ANCHORS_V3, [1,1,1])

    def layers(self):
        return [self.back,    self.head.b1, self.neck[2], 
                self.head.c1, self.head.b2, self.neck[1], 
                self.head.c2, self.head.b3, self.neck[0]]
    
    def forward(self, x, targets=None):
        p8, p16, p32 = self.head(*self.back(x))
        p8, p16, p32 = map(lambda x, n: n(x), [p8, p16, p32], self.neck)
        return self.yolo(p8, p16, p32, targets=targets)
    
class Yolov3Tiny(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.back = BackboneV3Tiny()
        self.head = HeadV3Tiny(1024)
        self.neck = nn.ModuleList([nn.Sequential(Conv(c1, c2, 3, act=actV3), nn.Conv2d(c2, (nclasses+5)*3, 1)) for c1, c2 in [(384, 256), (256, 512)]])
        self.yolo = DetectV3(nclasses, [16,32], ANCHORS_V3_TINY, [1,1])

    def layers(self):
        return [self.back, self.head.b1, self.neck[1], self.head.c2, self.neck[0]]
    
    def forward(self, x, targets=None):
        p16, p32 = self.head(*self.back(x))
        p16, p32 = map(lambda x, n: n(x), [p16, p32], self.neck)
        return self.yolo(p16, p32, targets=targets)
    
class Yolov4(nn.Module):
    def __init__(self, nclasses, act=actV4):
        super().__init__()
        self.back = BackboneV4(act)
        self.head = HeadV4(actV3)
        self.neck = nn.ModuleList([nn.Sequential(Conv(c1, c2, 3, act=actV3), nn.Conv2d(c2, (nclasses+5)*3, 1))  for c1, c2 in [(128, 256), (256, 512), (512, 1024)]])
        self.yolo = DetectV3(nclasses, [8,16,32], ANCHORS_V4, [1.2, 1.1, 1.05])

    def layers(self):
        return [self.back, self.head.b1, self.head.c1, self.head.b2, 
                self.head.c2, self.head.b3, self.neck[0],
                self.head.c3, self.head.b4, self.neck[1],
                self.head.c4, self.head.b5, self.neck[2]]
    
    def forward(self, x, targets=None):
        p8, p16, p32 = self.head(*self.back(x))
        p8, p16, p32 = map(lambda x, n: n(x), [p8, p16, p32], self.neck)
        return self.yolo(p8, p16, p32, targets=targets)

class Yolov4Tiny(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.back = BackboneV4Tiny()
        self.head = HeadV3Tiny(512)
        self.neck = nn.ModuleList([nn.Sequential(Conv(c1, c2, 3, act=actV3), nn.Conv2d(c2, (nclasses+5)*3, 1)) for c1, c2 in [(384, 256), (256, 512)]])
        self.yolo = DetectV3(nclasses, [16,32], ANCHORS_V3_TINY, [1.05,1.5])

    def layers(self):
        return [self.back, self.head.b1, self.neck[1], self.head.c2, self.neck[0]]
    
    def forward(self, x, targets=None):
        p16, p32 = self.head(*self.back(x))
        p16, p32 = map(lambda x, n: n(x), [p16, p32], self.neck)
        return self.yolo(p16, p32, targets=targets)

class Yolov7(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        c3          = (nclasses+5)*3
        self.back   = BackboneV7()
        self.head   = HeadV7()
        self.neck1  = nn.ModuleList([RepConv(c1, c2) for c1, c2 in [(128,256), (256,512), (512,1024)]])
        self.neck2  = nn.ModuleList([nn.Conv2d(c2, c3, 1) for c2 in [256, 512, 1024]])
        self.yolo   = DetectV3(nclasses, [8,16,32], ANCHORS_V7, [1,1,1], is_v7=True)
       
    def forward(self, x, targets=None):
        p8, p16, p32 = self.head(*self.back(x))
        p8, p16, p32 = map(lambda x, n1, n2: n2(n1(x)), [p8, p16, p32], self.neck1, self.neck2)
        return self.yolo(p8, p16, p32, targets=targets)

class Yolov5(nn.Module):
    def __init__(self, variant, num_classes):
        super().__init__()
        self.v    = variant
        d, w, r   = get_variant_multiplesV5(variant)
        self.net  = BackboneV5(w, r, d)
        self.fpn  = HeadV5(w, r, d)
        self.head = Detect(num_classes, ch=(int(256*w), int(512*w), int(512*w*2)))

    def forward(self, x, labels=None):
        x = self.net(x)
        x = self.fpn(*x)
        return self.head(x, labels=labels)

class Yolov8(nn.Module):
    def __init__(self, variant, num_classes):
        super().__init__()
        self.v    = variant
        d, w, r   = get_variant_multiplesV8(variant)
        self.net  = BackboneV8(w, r, d)
        self.fpn  = HeadV8(w, r, d)
        self.head = Detect(num_classes, ch=(int(256*w), int(512*w), int(512*w*r)))

    def forward(self, x, labels=None):
        x = self.net(x)
        x = self.fpn(*x)
        return self.head(x, labels=labels)

def load_from_ultralytics(net: Union[Yolov5, Yolov8]):
    from ultralytics import YOLO
    import numpy as np

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

def load_darknet(net: Union[Yolov3, Yolov3Tiny], weights_path: str):
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

def load_yolov7(net: Yolov7, weights_pt: str):
    state1 = net.state_dict()
    state2 = torch.load(weights_pt, map_location='cpu')

    del state1['yolo.anchors']

    for anchor_key in [k for k in state2.keys() if 'anchor' in k]:
        del state2[anchor_key]

    assert (nP1 := len(state1)) == (nP2 := len(state2)), f"{nP1} != {nP2}"

    for p1, p2 in zip(state1.values(), state2.values(), strict=True):
        p1.data.copy_(p2.data)

    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03

@torch.no_grad()
def nms(preds: torch.Tensor, conf_thresh: float, nms_thresh: float , has_objectness: bool):
    def conf(): return preds[...,4] if has_objectness else preds[...,4:].max(-1)[0]
    idxs    = torch.where(conf() > conf_thresh)
    batch   = idxs[0]
    preds   = preds[idxs]
    nms     = torchvision.ops.batched_nms(preds[:,:4], conf(), batch, iou_threshold=nms_thresh)
    preds   = preds[nms]
    batch   = batch[nms]
    return batch, preds