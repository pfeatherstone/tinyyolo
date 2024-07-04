from   typing import Union
from   copy import deepcopy
from   functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from   torch.utils.cpp_extension import load
from   einops import rearrange, repeat, pack, unpack

assigner = load(name="assigner", sources=["assigner.cpp"], extra_cflags=["-O3", "-ffast-math", "-march=native", "-std=c++20"], verbose=True)

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

def get_variant_multiplesV5(variant: str):
    match variant:
        case 'n': return (0.33, 0.25, 2.0)
        case 's': return (0.33, 0.50, 2.0)
        case 'm': return (0.67, 0.75, 2.0) 
        case 'l': return (1.00, 1.00, 2.0) 
        case 'x': return (1.33, 1.25, 2.0)

def get_variant_multiplesV8(variant: str):
    match variant:
        case 'n': return (0.33, 0.25, 2.0)
        case 's': return (0.33, 0.50, 2.0)
        case 'm': return (0.67, 0.75, 1.5)
        case 'l': return (1.00, 1.00, 1.0)
        case 'x': return (1.00, 1.25, 1.0)

def get_variant_multiplesV10(variant: str):
    match variant:
        case 'n': return (0.33, 0.25, 2.0)
        case 's': return (0.33, 0.50, 2.0)
        case 'm': return (0.67, 0.75, 1.5)
        case 'b': return (0.67, 1.00, 1.0)
        case 'l': return (1.00, 1.00, 1.0)
        case 'x': return (1.00, 1.25, 1.0)

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
        m1.eps      = m2.eps
        m1.momentum = m2.momentum
   
def init_batchnorms(net: nn.Module):
    for m in batchnorms(net):
        m.eps = 1e-3
        m.momentum = 0.03

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

def Conv(c1, c2, k=1, s=1, p=None, g=None, act=nn.SiLU(True)):
    return nn.Sequential(nn.Conv2d(c1, c2, k, s, default(p,k//2), groups=default(g,1), bias=False),
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

def Bottleneck(c1, c2=None, k=(3, 3), shortcut=True, g=1, e=0.5, act=nn.SiLU(True)):
    c2  = default(c2, c1)
    c_  = int(c2 * e)
    net = nn.Sequential(Conv(c1, c_, k[0], act=act), Conv(c_, c2, k[1], g=g, act=act))
    return Residual(net) if shortcut else net

def SCDown(c1, c2, k, s):
    return nn.Sequential(Conv(c1, c2, 1, 1), Conv(c2, c2, k=k, s=s, g=c2, act=nn.Identity()))
    
def MaxPool(stride):
    return nn.Sequential(nn.ZeroPad2d((0,1,0,1)), nn.MaxPool2d(kernel_size=2, stride=stride))

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_       = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)f
        self.m   = Repeat(Bottleneck(c_, shortcut=shortcut, k=(1, 3), e=1.0), n)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c_  = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)  # optional act=FReLU(c2)
        self.m   = nn.ModuleList(Bottleneck(self.c_, k=(3, 3), e=1.0, g=g, shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class CspBlock(nn.Module):
    def __init__(self, c1, c2, f=1, e=1, act=actV3, n=1):
        super().__init__()
        conv    = partial(Conv, act=act)
        c_      = int(c2 * f)  # hidden channels
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
        x = self.c4(torch.cat([b, a], 1))
        return x
    
class RepVGGDW(torch.nn.Module):
    def __init__(self, ed):
        super().__init__()
        self.conv   = Conv(ed, ed, 7, g=ed, act=nn.Identity())
        self.conv1  = Conv(ed, ed, 3, g=ed, act=nn.Identity())
        self.dim    = ed
    
    def forward(self, x):
        return F.silu(self.conv(x) + self.conv1(x), inplace=True)

class RepConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = Conv(c1, c2, 3, act=nn.Identity())
        self.c2 = Conv(c1, c2, 1, act=nn.Identity())
        self.bn = nn.BatchNorm2d(c1) if c1==c2 else None
    def forward(self, x):
        id_out = self.bn(x) if exists(self.bn) else 0
        return F.silu(self.c1(x) + self.c2(x) + id_out, inplace=True)

def CIB(c1, c2, shortcut=True, e=0.5, lk=False):
    c_  = int(c2 * e)  # hidden channels
    net = nn.Sequential(Conv(c1, c1, 3, g=c1),
                        Conv(c1, 2 * c_, 1),
                        Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
                        Conv(2 * c_, c2, 1),
                        Conv(c2, c2, 3, g=c2))
    return Residual(net) if shortcut else net
    
def C2fCIB(c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
    net   = C2f(c1, c2, n, shortcut, g, e)
    net.m = nn.ModuleList(CIB(net.c_, net.c_, shortcut, e=1.0, lk=lk) for _ in range(n))
    return net

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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads  = num_heads
        self.dim_head   = dim // num_heads
        self.key_dim    = int(self.dim_head * attn_ratio)
        h               = (self.dim_head + self.key_dim*2) * num_heads
        self.qkv        = Conv(dim, h, 1, act=nn.Identity())
        self.proj       = Conv(dim, dim, 1, act=nn.Identity())
        self.pe         = Conv(dim, dim, 3, 1, g=dim, act=nn.Identity())

    def forward(self, x):
        H, W    = x.shape[-2:]
        q, k, v = rearrange(self.qkv(x), 'b (h d) y x -> b h (y x) d', h=self.num_heads).split([self.key_dim, self.key_dim, self.dim_head], -1)
        x       = F.scaled_dot_product_attention(q, k, v)
        x, v    = map(lambda t: rearrange(t, 'b h (y x) d -> b (h d) y x', y=H, x=W), (x, v))
        x       = self.proj(x + self.pe(v))
        return x

class PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert(c1 == c2)
        c_       = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * c_, 1)
        self.cv2 = Conv(2 * c_, c1, 1)
        self.net = nn.Sequential(Residual(Attention(c_, attn_ratio=0.5, num_heads=c_ // 64)),
                                 Residual(nn.Sequential(Conv(c_, c_*2, 1), Conv(c_*2, c_, 1, act=nn.Identity()))))
        
    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((a, self.net(b)), 1))

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

class BackboneV10(nn.Module):
    def __init__(self, w, r, d, variant):
        super().__init__()
        self.b0 = Conv(c1=3, c2= int(64*w), k=3, s=2)
        self.b1 = Conv(int(64*w), int(128*w), k=3, s=2)
        self.b2 = C2f(c1=int(128*w), c2=int(128*w), n=round(3*d), shortcut=True)
        self.b3 = Conv(int(128*w), int(256*w), k=3, s=2)
        self.b4 = C2f(c1=int(256*w), c2=int(256*w), n=round(6*d), shortcut=True)
        self.b5 = SCDown(int(256*w), int(512*w), k=3, s=2)
        match variant:
            case 'x': self.b6 = C2fCIB(c1=int(512*w), c2=int(512*w), n=round(6*d), shortcut=True)
            case _  : self.b6 = C2f(c1=int(512*w), c2=int(512*w), n=round(6*d), shortcut=True)
        self.b7 = SCDown(int(512*w), int(512*w*r), k=3, s=2)
        match variant:
            case 'n': self.b8 = C2f(c1=int(512*w*r), c2=int(512*w*r), n=round(3*d), shortcut=True)
            case 's': self.b8 = C2fCIB(c1=int(512*w*r), c2=int(512*w*r), n=round(3*d), shortcut=True, lk=True)
            case _  : self.b8 = C2fCIB(c1=int(512*w*r), c2=int(512*w*r), n=round(3*d), shortcut=True, lk=False)
        self.b9 = SPPF(int(512*w*r), int(512*w*r))
        self.b10 = PSA(int(512*w*r), int(512*w*r))

    def forward(self, x):
        x4  = self.b4(self.b3(self.b2(self.b1(self.b0(x))))) # 4  P3/8
        x6  = self.b6(self.b5(x4))                           # 6  P4/16
        x10 = self.b10(self.b9(self.b8(self.b7(x6))))        # 10 P5/32
        return x4, x6, x10

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
    def __init__(self, w, r, d):
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

class HeadV10(nn.Module):
    def __init__(self, w, r, d, variant):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        match variant:
            case 'n'|'s'|'m': self.n1 = C2f(c1=int(512*w*(1+r)), c2=int(512*w), n=round(3*d))
            case _          : self.n1 = C2fCIB(c1=int(512*w*(1+r)), c2=int(512*w), n=round(3*d), shortcut=True)
        self.n2 = C2f(c1=int(768*w), c2=int(256*w), n=round(3*d))
        self.n3 = Conv(c1=int(256*w), c2=int(256*w), k=3, s=2)
        match variant:
            case 'n'|'s': self.n4 = C2f(c1=int(768*w), c2=int(512*w), n=round(3*d))
            case _      : self.n4 = C2fCIB(c1=int(768*w), c2=int(512*w), n=round(3*d), shortcut=True)
        self.n5 = SCDown(c1=int(512* w), c2=int(512 * w), k=3, s=2)
        match variant:
            case 'n'|'s': self.n6 = C2fCIB(c1=int(512*w*(1+r)), c2=int(512*w*r), n=round(3*d), shortcut=True, lk=True)
            case _      : self.n6 = C2fCIB(c1=int(512*w*(1+r)), c2=int(512*w*r), n=round(3*d), shortcut=True, lk=False)

    def forward(self, x4, x6, x10):
        x13 = self.n1(torch.cat([self.up(x10),x6], 1))  # 13
        x16 = self.n2(torch.cat([self.up(x13),x4], 1))  # 16 (P3/8-small)
        x19 = self.n4(torch.cat([self.n3(x16),x13], 1)) # 19 (P4/16-medium)
        x22 = self.n6(torch.cat([self.n5(x19),x10], 1)) # 22 (P5/32-large)
        return [x16, x19, x22]

def dfl_loss (
    target_bbox,        # [B,N,4] (input resolution)
    target_mask,        # [B,N]
    target_scores_sum,  # [0]
    sxy,                # [N,2] (input resolution)
    strides,            # [N]
    pred_dists          # [B,N,4,reg_max]
) :
    # Bring back to feature pyramid level resolution
    target_bbox = target_bbox / strides
    sxy         = sxy / strides

    # Distance
    reg_max      = pred_dists.shape[-1]
    lt, rb       = target_bbox.chunk(2,2)
    target_dists = torch.cat((sxy - lt, rb - sxy), -1).clamp_(0, reg_max - 1 - 0.01)
    target_dists = target_dists[target_mask]
    pred_dists   = pred_dists[target_mask].view(-1, reg_max)

    tl = target_dists.long()    # target left
    tr = tl + 1                 # target right
    wl = tr - target_dists      # weight left
    wr = 1 - wl                 # weight right
    loss_dfl = (
        F.cross_entropy(pred_dists, tl.view(-1), reduction="none").view(tl.shape) * wl + 
        F.cross_entropy(pred_dists, tr.view(-1), reduction="none").view(tl.shape) * wr
    ).mean(-1, keepdim=True).sum() / target_scores_sum

    return loss_dfl
    
class DetectV3(nn.Module):
    def __init__(self, nc, strides, anchors, scales, ch, is_v7=False):
        super().__init__()
        self.register_buffer('anchors_wh', torch.tensor(anchors).float())
        conv            = RepConv if is_v7 else partial(Conv, k=3, act=actV3)
        self.cv         = nn.ModuleList([nn.Sequential(conv(c1, c2), nn.Conv2d(c2, (nc+5)*3, 1)) for c1, c2 in ch])
        self.nc         = nc
        self.strides    = strides
        self.scales     = scales
        self.v7         = is_v7

    @torch.no_grad()
    def make_anchors(self, feats):
        xys, awhs, strides, scales = [], [], [], []
        for x, awh , stride, scale in zip(feats, self.anchors_wh, self.strides, self.scales):
            a, h, w = awh.shape[0], x.shape[2], x.shape[3]
            sx      = (torch.arange(end=w, device=x.device) + 0.5) * stride
            sy      = (torch.arange(end=h, device=x.device) + 0.5) * stride
            sy, sx  = torch.meshgrid(sy, sx, indexing='ij')
            xy      = repeat([sx,sy], 'c h w -> (a h w) c', a=a)
            awh     = repeat(awh, 'a c -> (a h w) c', h=h, w=w)
            xys.append(xy)
            awhs.append(awh)
            strides.append(torch.full((a*h*w,1), fill_value=stride, device=x.device))
            scales.append(torch.full((a*h*w,1), fill_value=scale, device=x.device))
        return *pack(xys, '* c'), torch.cat(awhs,0), torch.cat(strides,0), torch.cat(scales,0)

    def to_wh(self, wh, awh):
        match self.v7:
            case True:  return awh * (wh.sigmoid() ** 2) * 4
            case False: return awh * wh.exp()
    
    def forward(self, xs, targets=None):
        sxy, ps, awh, strides, scales = self.make_anchors(xs)
        feats           = [rearrange(n(x), 'b (a f) h w -> b (a h w) f', f=self.nc+5) for x,n in zip(xs, self.cv)]
        xy, wh, l, cls  = torch.cat(feats,1).split((2,2,1,self.nc), -1) 
        xy              = strides * ((xy.sigmoid() - 0.5) * scales) + sxy
        wh              = self.to_wh(wh, awh)
        box             = torch.cat([xy-wh/2, xy+wh/2], -1)
        pred            = torch.cat([box, l.sigmoid(), cls.sigmoid()], -1)

        if exists(targets):
            anchors               = torch.cat([sxy-awh/2, sxy+awh/2],-1)
            tboxes, tscores, tcls = assigner.atss(anchors, targets, [p[0] for p in ps], self.nc, 9)
            mask                  = tscores > 0

            # CIOU loss (positive samples)
            tgt_scores_sum = max(tscores.sum(), 1)
            weight   = tscores[mask]
            loss_iou = (torchvision.ops.complete_box_iou_loss(box[mask], tboxes[mask], reduction='none') * weight).sum() / tgt_scores_sum
            
            # Class loss (positive samples)
            loss_cls = (F.binary_cross_entropy_with_logits(cls[mask], tcls[mask], reduction='none') * weight.unsqueeze(-1)).sum() / tgt_scores_sum
            
            # Objectness loss (positive + negative samples)
            l_obj       = l[mask].squeeze(-1)
            l_noobj     = l[~mask].squeeze(-1)
            loss_obj    = F.binary_cross_entropy_with_logits(l_obj, tscores[mask], reduction='none').sum() / tgt_scores_sum
            loss_noobj  = F.binary_cross_entropy_with_logits(l_noobj, torch.zeros_like(l_noobj), reduction='mean')

        return pred if not exists(targets) else (pred, {'iou': loss_iou, 'cls': loss_cls, 'obj': loss_obj, 'noobj': loss_noobj})

class Detect(nn.Module):
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc         = nc                        # number of classes
        self.reg_max    = 16                        # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no         = nc + self.reg_max * 4     # number of outputs per anchor
        self.strides    = [8, 16, 32]               # strides computed during build
        self.c2         = max((16, ch[0] // 4, self.reg_max * 4))
        self.c3         = max(ch[0], min(self.nc, 100))  # channels
        self.cv2        = nn.ModuleList(nn.Sequential(Conv(x, self.c2, 3), Conv(self.c2, self.c2, 3), nn.Conv2d(self.c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3        = nn.ModuleList(nn.Sequential(Conv(x, self.c3, 3), Conv(self.c3, self.c3, 3), nn.Conv2d(self.c3, self.nc, 1)) for x in ch)
        self.r          = nn.Parameter(torch.arange(self.reg_max).float(), requires_grad=False)

    @torch.no_grad()
    def make_anchors(self, feats):
        xys, strides = [], []
        for x, stride in zip(feats, self.strides):
            h, w    = x.shape[2], x.shape[3]
            sx      = (torch.arange(end=w, device=x.device) + 0.5) * stride
            sy      = (torch.arange(end=h, device=x.device) + 0.5) * stride
            sy, sx  = torch.meshgrid(sy, sx, indexing='ij')
            xy      = rearrange([sx,sy], 'c h w -> (h w) c')
            xys.append(xy)
            strides.append(torch.full((h*w,1), fill_value=stride, device=x.device))
        return *pack(xys, '* c'), torch.cat(strides,0)
    
    def forward(self, xs, targets=None):
        sxy, ps, strides = self.make_anchors(xs)
        feats       = [rearrange(torch.cat((c1(x), c2(x)), 1), 'b f h w -> b (h w) f') for x,c1,c2 in zip(xs, self.cv2, self.cv3)]
        dist, cls   = torch.cat(feats, 1).split((4 * self.reg_max, self.nc), -1)
        dist        = rearrange(dist, 'b n (k r) -> b n k r', k=4)
        lt, rb      = torch.einsum('bnkr, r -> bnk', dist.softmax(-1), self.r).chunk(2, 2)
        x1y1        = sxy - lt*strides
        x2y2        = sxy + rb*strides
        box         = torch.cat([x1y1,x2y2],-1)
        pred        = torch.cat((box, cls.sigmoid()), -1)

        if exists(targets):
            # awh                     = torch.full_like(sxy, fill_value=5.0) * strides # Fake height and width for the sake of ATSS
            # anchors                 = torch.cat([sxy-awh/2, sxy+awh/2],-1)
            # tboxes, tscores, tcls   = assigner.atss(anchors, targets, [p[0] for p in ps], self.nc, 9)
            tboxes, tscores, tcls   = assigner.tal(box, cls.sigmoid(), sxy, targets, 9, 0.5, 6.0)
            mask                    = tscores > 0

            # CIOU loss (positive samples)
            tgt_scores_sum = max(tscores.sum(), 1)
            weight   = tscores[mask]
            loss_iou = (torchvision.ops.complete_box_iou_loss(box[mask], tboxes[mask], reduction='none') * weight).sum() / tgt_scores_sum

            # DFL loss (positive samples)
            loss_dfl = dfl_loss(tboxes, mask, tgt_scores_sum, sxy, strides, dist)
            
            # Class loss (positive samples + negative)
            loss_cls = F.binary_cross_entropy_with_logits(cls, tcls*tscores.unsqueeze(-1), reduction='none').sum() / tgt_scores_sum

        return pred if not exists(targets) else (pred, {'iou': loss_iou, 'dfl': loss_dfl, 'cls': loss_cls})
        
class DetectV10(Detect):
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, x, 3, g=x), 
                                               Conv(x, self.c3, 1),
                                               Conv(self.c3, self.c3, 3, g=self.c3), 
                                               Conv(self.c3, self.c3, 1),
                                               nn.Conv2d(self.c3, self.nc, 1)) for x in ch)

        self.one2one_cv2 = deepcopy(self.cv2)
        self.one2one_cv3 = deepcopy(self.cv3)
        self.max_det = 100
    
    def forward(self, x):
        # TODO: implement all the topk stuff. I think yolov10 doesn't need NMS. But you can you use it in inference mode for now.
        return self.inference(self.forward_feat(x, self.one2one_cv2, self.one2one_cv3))
    
class Yolov3(nn.Module):
    def __init__(self, nclasses, spp):
        super().__init__()
        self.net  = Darknet53()
        self.fpn  = HeadV3(spp)
        self.head = DetectV3(nclasses, [8,16,32], ANCHORS_V3, [1,1,1], ch=[(128, 256), (256, 512), (512, 1024)])

    def layers(self):
        return [self.net,    self.fpn.b1, self.head.cv[2], 
                self.fpn.c1, self.fpn.b2, self.head.cv[1], 
                self.fpn.c2, self.fpn.b3, self.head.cv[0]]
    
    def forward(self, x, targets=None):
        x = self.net(x)
        x = self.fpn(*x)
        return self.head(x, targets=targets)
    
class Yolov3Tiny(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.net  = BackboneV3Tiny()
        self.fpn  = HeadV3Tiny(1024)
        self.head = DetectV3(nclasses, [16,32], ANCHORS_V3_TINY, [1,1], ch=[(384, 256), (256, 512)])

    def layers(self):
        return [self.net, self.fpn.b1, self.head.cv[1], self.fpn.c2, self.head.cv[0]]
    
    def forward(self, x, targets=None):
        x = self.net(x)
        x = self.fpn(*x)
        return self.head(x, targets=targets)
    
class Yolov4(nn.Module):
    def __init__(self, nclasses, act=actV4):
        super().__init__()
        self.net  = BackboneV4(act)
        self.fpn  = HeadV4(actV3)
        self.head = DetectV3(nclasses, [8,16,32], ANCHORS_V4, [1.2, 1.1, 1.05], ch=[(128, 256), (256, 512), (512, 1024)])

    def layers(self):
        return [self.net, self.fpn.b1, self.fpn.c1, self.fpn.b2, 
                self.fpn.c2, self.fpn.b3, self.head.cv[0],
                self.fpn.c3, self.fpn.b4, self.head.cv[1],
                self.fpn.c4, self.fpn.b5, self.head.cv[2]]
    
    def forward(self, x, targets=None):
        x = self.net(x)
        x = self.fpn(*x)
        return self.head(x, targets=targets)

class Yolov4Tiny(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.net  = BackboneV4Tiny()
        self.fpn  = HeadV3Tiny(512)
        self.head = DetectV3(nclasses, [16,32], ANCHORS_V3_TINY, [1.05,1.5], ch=[(384, 256), (256, 512)])

    def layers(self):
        return [self.net, self.fpn.b1, self.head.cv[1], self.fpn.c2, self.head.cv[0]]
    
    def forward(self, x, targets=None):
        x = self.net(x)
        x = self.fpn(*x)
        return self.head(x, targets=targets)

class Yolov7(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        ch = [(128,256), (256,512), (512,1024)]
        self.net  = BackboneV7()
        self.fpn  = HeadV7()
        self.head = DetectV3(nclasses, [8,16,32], ANCHORS_V7, [2,2,2], ch=ch, is_v7=True)

    def layers(self):
        return [self.net, self.fpn, self.head.cv[0][0], self.head.cv[1][0], self.head.cv[2][0], self.head.cv[0][1], self.head.cv[1][1], self.head.cv[2][1]]
       
    def forward(self, x, targets=None):
        x = self.net(x)
        x = self.fpn(*x)
        return self.head(x, targets=targets)

class Yolov5(nn.Module):
    def __init__(self, variant, num_classes):
        super().__init__()
        self.v    = variant
        d, w, r   = get_variant_multiplesV5(variant)
        self.net  = BackboneV5(w, r, d)
        self.fpn  = HeadV5(w, r, d)
        self.head = Detect(num_classes, ch=(int(256*w), int(512*w), int(512*w*2)))

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(*x)
        return self.head(x)

class Yolov8(nn.Module):
    def __init__(self, variant, num_classes):
        super().__init__()
        self.v    = variant
        d, w, r   = get_variant_multiplesV8(variant)
        self.net  = BackboneV8(w, r, d)
        self.fpn  = HeadV8(w, r, d)
        self.head = Detect(num_classes, ch=(int(256*w), int(512*w), int(512*w*r)))

    def forward(self, x, targets=None):
        x = self.net(x)
        x = self.fpn(*x)
        return self.head(x, targets)

class Yolov10(nn.Module):
    def __init__(self, variant, num_classes):
        super().__init__()
        self.v    = variant
        d, w, r   = get_variant_multiplesV10(variant)
        self.net  = BackboneV10(w, r, d, variant)
        self.fpn  = HeadV10(w, r, d, variant)
        self.head = DetectV10(num_classes, ch=(int(256*w), int(512*w), int(512*w*r)))

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(*x)
        return self.head(x)

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
    
def load_darknet(net: Union[Yolov3, Yolov3Tiny, Yolov4, Yolov4Tiny], weights_path: str):
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
    def params1():
        for l in net.layers():
            for k, v in l.state_dict().items():
                if 'anchor' not in k:
                    yield v
    
    def params2():
        state2 = torch.load(weights_pt, map_location='cpu')
        for k, v in state2.items():
            if 'anchor' not in k:
                yield v

    for p1, p2 in zip(params1(), params2(), strict=True):
        p1.data.copy_(p2.data)

    init_batchnorms(net)

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