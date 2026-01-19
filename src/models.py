from   copy import deepcopy
from   functools import partial
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

actV3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
actV4 = nn.Mish(inplace=True)

def get_variant_multiplesV5(variant: str):
    match variant:
        case 'n': return (0.33, 0.25, 2.0)
        case 's': return (0.33, 0.50, 2.0)
        case 'm': return (0.67, 0.75, 2.0) 
        case 'l': return (1.00, 1.00, 2.0) 
        case 'x': return (1.33, 1.25, 2.0)

def get_variant_multiplesV6(variant: str):
    # depth, width, csp, csp_e, distill
    match variant:
        case 'n': return (0.33, 0.25, False, 0,   True)
        case 's': return (0.33, 0.50, False, 0,   True)
        case 'm': return (0.60, 0.75, True,  2/3, False)
        # case 'l': return (1.00, 1.00)

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

def get_variant_multiplesV11(variant: str):
    match variant:
        case 'n': return (0.50, 0.25, 2.0)
        case 's': return (0.50, 0.50, 2.0)
        case 'm': return (0.50, 1.00, 1.0)
        case 'l': return (1.00, 1.00, 1.0)
        case 'x': return (1.00, 1.50, 1.0)

def get_variant_multiplesV12(variant: str):
    return get_variant_multiplesV11(variant)

def get_variant_multiplesV26(variant: str):
    return get_variant_multiplesV11(variant)

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

def count_parameters(net: torch.nn.Module, include_stats=True):
    return sum(p.numel() for p in net.parameters()) + (sum(m.running_mean.numel() + m.running_var.numel() for m in batchnorms(net)) if include_stats else 0)
        
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def Repeat(module, N):
    return nn.Sequential(*[deepcopy(module) for _ in range(N)])

class Residual(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self, x):
        return x + self.f(x)

class Conv(nn.Sequential):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=nn.SiLU(True)):
        super().__init__(nn.Conv2d(c1, c2, k, s, default(p,k//2), groups=g, bias=False),
                         nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03),
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

def SCDown(c1, c2, k, s):
    return nn.Sequential(Conv(c1, c2, 1, 1), Conv(c2, c2, k=k, s=s, g=c2, act=nn.Identity()))
    
def MaxPool(stride):
    return nn.Sequential(nn.ZeroPad2d((0,1,0,1)), nn.MaxPool2d(kernel_size=2, stride=stride))
    
class RepVGGDW(torch.nn.Module):
    def __init__(self, ed):
        super().__init__()
        self.conv   = Conv(ed, ed, 7, g=ed, act=nn.Identity())
        self.conv1  = Conv(ed, ed, 3, g=ed, act=nn.Identity())
        self.dim    = ed
    
    def forward(self, x):
        return F.silu(self.conv(x) + self.conv1(x), inplace=True)

class RepConv(nn.Module):
    def __init__(self, c1, c2, s=1, act=F.silu):
        super().__init__()
        self.c1 = Conv(c1, c2, k=3, s=s, act=nn.Identity())
        self.c2 = Conv(c1, c2, k=1, s=s, act=nn.Identity())
        self.bn = nn.BatchNorm2d(c1) if c1==c2 and s==1 else None
        self.act = act

    def forward(self, x):
        id_out = self.bn(x) if exists(self.bn) else 0 
        return self.act(self.c1(x) + self.c2(x) + id_out, inplace=True)

class BottleRep(nn.Module):
    def __init__(self, c1, c2, weight=False):
        super().__init__()
        self.conv1 = RepConv(c1, c2, act=F.relu)
        self.conv2 = RepConv(c1, c2, act=F.relu)
        self.add   = c1==c2
        self.alpha = nn.Parameter(torch.ones(1)) if weight else 1.0

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return (y + self.alpha * x) if self.add else y

def RepBlock(c1, c2, n=1):
    block = partial(RepConv, act=F.relu)
    return nn.Sequential(block(c1, c2), *[block(c2, c2) for _ in range(n - 1)])

def BottleRepBlock(c1, c2, n=1):
    n = n // 2
    block = partial(BottleRep, weight=True)
    return nn.Sequential(block(c1, c2), *[block(c2, c2) for _ in range(n - 1)])
        
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c_  = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c_, 1)
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)
        self.m   = nn.ModuleList(Bottleneck(self.c_, k=(3, 3), e=1.0, shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, k=(1,3), e=0.5):
        super().__init__()
        c_       = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.cv3 = Conv(2*c_, c2, 1)  # optional act=FReLU(c2)f
        self.m   = Repeat(Bottleneck(c_, shortcut=shortcut, k=k, e=1.0), n)

    def forward(self, x):
        a = self.cv1(x)
        b = self.m(self.cv2(x))
        return self.cv3(torch.cat((b, a), 1))

def C3k2(c1, c2, n=1, shortcut=True, e=0.5, c3k=False, attn=False):
    net = C2f(c1, c2, n, shortcut, e)
    if   attn: blk = nn.Sequential(Bottleneck(net.c_, k=(3,3), shortcut=shortcut), PSABlock(net.c_, num_heads=net.c_//64, attn_ratio=0.5))
    elif c3k:  blk = C3(net.c_, net.c_, k=(3,3), shortcut=shortcut, n=2)
    else:      blk = Bottleneck(net.c_, k=(3,3), shortcut=shortcut)
    net.m = nn.ModuleList(deepcopy(blk) for _ in range(n))
    return net

class C4(nn.Module):
    def __init__(self, c1, c2, f=1, e=1, act=actV3, n=1):
        super().__init__()
        conv     = partial(Conv, act=act)
        c_       = int(c2 * f)  # hidden channels
        self.cv1 = conv(c1, c_, 1)
        self.cv2 = conv(c1, c_, 1)
        self.cv3 = conv(c_, c_, 1)
        self.cv4 = conv(2*c_, c2, 1)
        self.m   = Repeat(Bottleneck(c_, k=(1,3), e=e, act=act), n)
        
    def forward(self, x):
        a = self.cv1(x)
        b = self.cv3(self.m(self.cv2(x)))
        x = self.cv4(torch.cat([b, a], 1))
        return x
    
class BepC3(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=nn.ReLU(True))
        self.cv2 = Conv(c1, c_, 1, 1, act=nn.ReLU(True))
        self.cv3 = Conv(2*c_, c2, 1, 1, act=nn.ReLU(True))
        self.m   = BottleRepBlock(c_, c_, n=n)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
    
def CIB(c1, c2, shortcut=True, e=0.5, lk=False):
    c_  = int(c2 * e)  # hidden channels
    net = nn.Sequential(Conv(c1, c1, 3, g=c1),
                        Conv(c1, 2 * c_, 1),
                        Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
                        Conv(2 * c_, c2, 1),
                        Conv(c2, c2, 3, g=c2))
    return Residual(net) if shortcut else net
    
def C2fCIB(c1, c2, n=1, shortcut=False, e=0.5, cib=True, lk=False):
    net   = C2f(c1, c2, n, shortcut, e)
    if cib: net.m = nn.ModuleList(CIB(net.c_, net.c_, shortcut, e=1.0, lk=lk) for _ in range(n))
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
    def __init__(self, c1, c2, acts=[nn.SiLU(True), nn.SiLU(True)], shortcut=False):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_          = c1 // 2  # hidden channels
        self.add    = shortcut and c1==c2
        self.cv1    = Conv(c1,   c_, 1, 1, act=acts[0])
        self.cv2    = Conv(c_*4, c2, 1, 1, act=acts[1])
        self.m      = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    def forward(self, input):
        x  = self.cv1(input)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y  = self.cv2(torch.cat((x, y1, y2, y3), 1))
        return input+y if self.add else y

class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, e=0.5, act=nn.SiLU(True)):
        super().__init__()
        c_   = int(2 * c2 * e)  # hidden channels
        conv = partial(Conv, act=act)
        self.cv1 = conv(c1, c_, 1, 1)
        self.cv2 = conv(c1, c_, 1, 1)
        self.cv3 = conv(c_, c_, 3, 1)
        self.cv4 = conv(c_, c_, 1, 1)
        self.m   = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.cv5 = conv(4 * c_, c_, 1, 1)
        self.cv6 = conv(c_, c_, 3, 1)
        self.cv7 = conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        y1 = self.m(x1)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y4 = self.cv6(self.cv5(torch.cat([x1, y1, y2, y3], 1)))
        return self.cv7(torch.cat((y0, y4), dim=1))

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5, area=None):
        super().__init__()
        self.num_heads  = num_heads
        self.dim_head   = dim // num_heads
        self.key_dim    = int(self.dim_head * attn_ratio)
        self.area       = default(area, 1)
        h               = (self.dim_head + self.key_dim*2) * num_heads
        k               = 7 if area else 3
        self.qkv        = Conv(dim, h, 1, act=nn.Identity())
        self.proj       = Conv(dim, dim, 1, act=nn.Identity())
        self.pe         = Conv(dim, dim, k, g=dim, act=nn.Identity())

    def forward(self, x):
        H, W    = x.shape[-2:]
        q, k, v = rearrange(self.qkv(x), 'b (h d) (a y) x -> (b a) h (y x) d', h=self.num_heads, a=self.area).split([self.key_dim, self.key_dim, self.dim_head], -1)
        x       = F.scaled_dot_product_attention(q, k, v)
        x, v    = map(lambda t: rearrange(t, '(b a) h (y x) d -> b (h d) (a y) x', x=W, a=self.area), (x, v))
        x       = self.proj(x + self.pe(v))
        return x

def PSABlock(c, num_heads=4, attn_ratio=0.5, area=None, e=2):
    c_ = int(c*e)
    return nn.Sequential(Residual(Attention(c, num_heads=num_heads, attn_ratio=attn_ratio, area=area)),
                         Residual(nn.Sequential(Conv(c, c_, 1), Conv(c_, c, 1, act=nn.Identity())))) 

class PSA(nn.Module):
    def __init__(self, c, e=0.5, n=1):
        super().__init__()
        c_       = int(c * e)  # hidden channels
        self.cv1 = Conv(c,    2*c_, 1)
        self.cv2 = Conv(2*c_, c,    1)
        self.net = Repeat(PSABlock(c_, num_heads=c_//64, attn_ratio=0.5), n)
        
    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((a, self.net(b)), 1))

class A2C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, a2=True, residual=False, area=1, mlp_ratio=2.0):
        super().__init__()
        self.c_  = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, self.c_, 1)
        self.cv2 = Conv((1 + n) * self.c_, c2, 1)
        self.g   = nn.Parameter(0.01 * torch.ones(c2)) if (a2 and residual) else None
        self.m   = nn.ModuleList(Repeat(PSABlock(self.c_, num_heads=self.c_//32, attn_ratio=1, area=area, e=mlp_ratio), 2) if a2 else
                                 C3(self.c_, self.c_, k=(3,3), shortcut=shortcut, n=2) for _ in range(n))

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        return x + y * self.g[None,:,None,None] if exists(self.g) else y

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        conv = partial(Conv, act=actV3)
        res  = partial(Bottleneck, act=actV3, k=(1, 3))
        self.stem = conv(3, 32, 3)
        self.b1   = nn.Sequential(conv( 32,   64, 3, 2), res(64))
        self.b2   = nn.Sequential(conv( 64,  128, 3, 2), Repeat(res(128), 2))
        self.b3   = nn.Sequential(conv(128,  256, 3, 2), Repeat(res(256), 8))
        self.b4   = nn.Sequential(conv(256,  512, 3, 2), Repeat(res(512), 8))
        self.b5   = nn.Sequential(conv(512, 1024, 3, 2), Repeat(res(1024),4))

    def forward(self, x):
        p8  = self.b3(self.b2(self.b1(self.stem(x))))
        p16 = self.b4(p8)
        p32 = self.b5(p16)
        return p8, p16, p32
    
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
        conv     = partial(Conv, act=act)
        c4       = partial(C4, act=act)
        self.b0  = conv(3, 32, 3)
        self.b1  = nn.Sequential(conv( 32,   64, 3, s=2), c4( 64,  64, f=1.0, e=0.5, n=1))
        self.b2  = nn.Sequential(conv( 64,  128, 3, s=2), c4(128, 128, f=0.5, e=1.0, n=2))
        self.b3  = nn.Sequential(conv(128,  256, 3, s=2), c4(256, 256, f=0.5, e=1.0, n=8))
        self.b4  = nn.Sequential(conv(256,  512, 3, s=2), c4(512, 512, f=0.5, e=1.0, n=8))
        self.b5  = nn.Sequential(conv(512, 1024, 3, s=2), c4(1024,1024,f=0.5, e=1.0, n=4))

    def forward(self, x):
        p8  = self.b3(self.b2(self.b1(self.b0(x))))
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
        self.b2 = nn.Sequential(BackboneV4TinyBlock(32),  MaxPool(2), conv(128, 128, 3, 1))
        self.b3 = nn.Sequential(BackboneV4TinyBlock(64),  MaxPool(2), conv(256, 256, 3, 1))
        self.b4 = nn.Sequential(BackboneV4TinyBlock(128), MaxPool(2), conv(512, 512, 3, 1))
    def forward(self, x):
        p8  = self.b2(self.b1(x))
        p16 = self.b3(p8)
        p32 = self.b4(p16)
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
        self.b1 = Conv(64, 128, 3, s=2)
        self.b2 = ElanBlock(128, mr=0.5, br=0.5, nb=2, nc=2)
        self.b3 = MaxPoolAndStrideConv(256)
        self.b4 = ElanBlock(256, mr=0.5, br=0.5, nb=2, nc=2)
        self.b5 = MaxPoolAndStrideConv(512)
        self.b6 = ElanBlock(512, mr=0.5, br=0.5, nb=2, nc=2)
        self.b7 = MaxPoolAndStrideConv(1024)
        self.b8 = ElanBlock(1024, mr=0.25, br=0.25, nb=2, nc=2, cout=1024)
        self.b9 = SPPCSPC(1024, 512)

    def forward(self, x):
        x4 = self.b4(self.b3(self.b2(self.b1(self.b0(x))))) # 4 P3/8
        x6 = self.b6(self.b5(x4))                           # 6 P4/16
        x9 = self.b9(self.b8(self.b7(x6)))                  # 9 P5/32
        return x4, x6, x9

class BackboneV5(nn.Module):
    def __init__(self, w, r, d):
        super().__init__()
        self.b0 = Conv(c1=3,            c2=int(64*w),    k=6, s=2, p=2)
        self.b1 = Conv(c1=int(64*w),    c2=int(128*w),   k=3, s=2)
        self.b2 = C3(c1=int(128*w),     c2=int(128*w),   n=round(3*d))
        self.b3 = Conv(c1=int(128*w),   c2=int(256*w),   k=3, s=2)
        self.b4 = C3(c1=int(256*w),     c2=int(256*w),   n=round(6*d))
        self.b5 = Conv(c1=int(256*w),   c2=int(512*w),   k=3, s=2)
        self.b6 = C3(c1=int(512*w),     c2=int(512*w),   n=round(9*d))
        self.b7 = Conv(c1=int(512*w),   c2=int(512*w*r), k=3, s=2)
        self.b8 = C3(c1=int(512*w*r),   c2=int(512*w*r), n=round(3*d))
        self.b9 = SPPF(c1=int(512*w*r), c2=int(512*w*r))

    def forward(self, x):
        x4 = self.b4(self.b3(self.b2(self.b1(self.b0(x))))) # 4 P3/8
        x6 = self.b6(self.b5(x4))                           # 6 P4/16
        x9 = self.b9(self.b8(self.b7(x6)))                  # 9 P5/32
        return x4, x6, x9

class BackboneV8(nn.Module):
    def __init__(self, w, r, d):
        super().__init__()
        self.b0 = Conv(c1=3,            c2=int(64*w),    k=3, s=2)
        self.b1 = Conv(c1=int(64*w),    c2=int(128*w),   k=3, s=2)
        self.b2 = C2f(c1=int(128*w),    c2=int(128*w),   n=round(3*d), shortcut=True)
        self.b3 = Conv(c1=int(128*w),   c2=int(256*w),   k=3, s=2)
        self.b4 = C2f(c1=int(256*w),    c2=int(256*w),   n=round(6*d), shortcut=True)
        self.b5 = Conv(c1=int(256*w),   c2=int(512*w),   k=3, s=2)
        self.b6 = C2f(c1=int(512*w),    c2=int(512*w),   n=round(6*d), shortcut=True)
        self.b7 = Conv(c1=int(512*w),   c2=int(512*w*r), k=3, s=2)
        self.b8 = C2f(c1=int(512*w*r),  c2=int(512*w*r), n=round(3*d), shortcut=True)
        self.b9 = SPPF(c1=int(512*w*r), c2=int(512*w*r))

    def forward(self, x):
        x4 = self.b4(self.b3(self.b2(self.b1(self.b0(x))))) # 4 P3/8
        x6 = self.b6(self.b5(x4))                           # 6 P4/16
        x9 = self.b9(self.b8(self.b7(x6)))                  # 9 P5/32
        return x4, x6, x9

class BackboneV10(nn.Module):
    def __init__(self, w, r, d, variant):
        super().__init__()
        self.b0 = Conv(c1=3,                c2= int(64*w),   k=3, s=2)
        self.b1 = Conv(c1=int(64*w),        c2=int(128*w),   k=3, s=2)
        self.b2 = C2f(c1=int(128*w),        c2=int(128*w),   n=round(3*d), shortcut=True)
        self.b3 = Conv(c1=int(128*w),       c2=int(256*w),   k=3, s=2)
        self.b4 = C2f(c1=int(256*w),        c2=int(256*w),   n=round(6*d), shortcut=True)
        self.b5 = SCDown(c1=int(256*w),     c2=int(512*w),   k=3, s=2)
        self.b6 = C2fCIB(c1=int(512*w),     c2=int(512*w),   n=round(6*d), shortcut=True, cib=variant=='x')
        self.b7 = SCDown(c1=int(512*w),     c2=int(512*w*r), k=3, s=2)
        self.b8 = C2fCIB(c1=int(512*w*r),   c2=int(512*w*r), n=round(3*d), shortcut=True, cib=not variant=='n', lk=variant=='s')
        self.b9 = SPPF(c1=int(512*w*r),     c2=int(512*w*r))
        self.b10 = PSA(int(512*w*r))

    def forward(self, x):
        x4  = self.b4(self.b3(self.b2(self.b1(self.b0(x))))) # 4  P3/8
        x6  = self.b6(self.b5(x4))                           # 6  P4/16
        x10 = self.b10(self.b9(self.b8(self.b7(x6))))        # 10 P5/32
        return x4, x6, x10

class BackboneV11(nn.Module):
    def __init__(self, w, r, d, variant, sppf_shortcut=False, sppf_acts=[nn.SiLU(True), nn.SiLU(True)]):
        super().__init__()
        c3k = variant in "mlx"
        self.b0 = Conv(c1=3,            c2=int(64*w),    k=3, s=2)
        self.b1 = Conv(c1=int(64*w),    c2=int(128*w),   k=3, s=2)
        self.b2 = C3k2(c1=int(128*w),   c2=int(256*w),   n=round(2*d), e=0.25, c3k=c3k)
        self.b3 = Conv(c1=int(256*w),   c2=int(256*w),   k=3, s=2)
        self.b4 = C3k2(c1=int(256*w),   c2=int(512*w),   n=round(2*d), e=0.25, c3k=c3k)
        self.b5 = Conv(c1=int(512*w),   c2=int(512*w),   k=3, s=2)
        self.b6 = C3k2(c1=int(512*w),   c2=int(512*w),   n=round(2*d), e=0.50, c3k=True)
        self.b7 = Conv(c1=int(512*w),   c2=int(512*w*r), k=3, s=2)
        self.b8 = C3k2(c1=int(512*w*r), c2=int(512*w*r), n=round(2*d), e=0.50, c3k=True)
        self.b9 = SPPF(c1=int(512*w*r), c2=int(512*w*r), shortcut=sppf_shortcut, acts=sppf_acts)
        self.b10 = PSA(int(512*w*r), n=round(2*d))

    def forward(self, x):
        x4  = self.b4(self.b3(self.b2(self.b1(self.b0(x))))) # 4 P3/8
        x6  = self.b6(self.b5(x4))                           # 6 P4/16
        x10 = self.b10(self.b9(self.b8(self.b7(x6))))        # 10 P5/32
        return x4, x6, x10

class BackboneV12(nn.Module):
    def __init__(self, w, r, d, variant):
        super().__init__()
        c3k, res, mlp = variant in "mlx", variant in "lx", 1.2 if variant in "lx" else 2.0
        self.b0 = Conv(c1=3,            c2=int(64*w),    k=3, s=2)
        self.b1 = Conv(c1=int(64*w),    c2=int(128*w),   k=3, s=2)
        self.b2 = C3k2(c1=int(128*w),   c2=int(256*w),   n=round(2*d), e=0.25, c3k=c3k)
        self.b3 = Conv(c1=int(256*w),   c2=int(256*w),   k=3, s=2)
        self.b4 = C3k2(c1=int(256*w),   c2=int(512*w),   n=round(2*d), e=0.25, c3k=c3k)
        self.b5 = Conv(c1=int(512*w),   c2=int(512*w),   k=3, s=2)
        self.b6 = A2C2f(c1=int(512*w),  c2=int(512*w),   n=round(4*d), a2=True, area=4, residual=res, mlp_ratio=mlp)
        self.b7 = Conv(c1=int(512*w),   c2=int(512*w*r), k=3, s=2)
        self.b8 = A2C2f(c1=int(512*w*r),c2=int(512*w*r), n=round(4*d), a2=True, area=1, residual=res, mlp_ratio=mlp)
    
    def forward(self, x):
        x4 = self.b4(self.b3(self.b2(self.b1(self.b0(x))))) # 4 P3/8
        x6 = self.b6(self.b5(x4))                           # 6 P4/16
        x8 = self.b8(self.b7(x6))                           # 8 P5/32
        return x4, x6, x8 

class EfficientRep(nn.Module):
    def __init__(self, w, d, cspsppf=False):
        super().__init__()
        sppf    = partial(SPPCSPC, e=0.25, act=nn.ReLU(True)) if cspsppf else partial(SPPF, acts=[nn.ReLU(True), nn.ReLU(True)])
        self.b0 = RepConv( c1=3,          c2=int(64*w),  s=2, act=F.relu)
        self.b1 = RepConv( c1=int(64*w),  c2=int(128*w), s=2, act=F.relu)
        self.b2 = RepBlock(c1=int(128*w), c2=int(128*w), n=round(6*d))
        self.b3 = RepConv( c1=int(128*w), c2=int(256*w), s=2, act=F.relu)
        self.b4 = RepBlock(c1=int(256*w), c2=int(256*w), n=round(12*d))
        self.b5 = RepConv( c1=int(256*w), c2=int(512*w), s=2, act=F.relu)
        self.b6 = RepBlock(c1=int(512*w), c2=int(512*w), n=round(18*d))
        self.b7 = RepConv( c1=int(512*w), c2=int(1024*w),s=2, act=F.relu)
        self.b8 = RepBlock(c1=int(1024*w),c2=int(1024*w),n=round(6*d))
        self.b9 = sppf(c1=int(1024*w),c2=int(1024*w))

    def forward(self, x):
        x4  = self.b2(self.b1(self.b0(x)))      # p2/4
        x8  = self.b4(self.b3(x4))              # p3/8
        x16 = self.b6(self.b5(x8))              # p4/16 
        x32 = self.b9(self.b8(self.b7(x16)))    # p5/32
        return x4, x8, x16, x32
    
class CSPBepBackbone(nn.Module):
    def __init__(self, w, d, csp_e=1/2, cspsppf=False):
        super().__init__()
        sppf    = partial(SPPCSPC, e=0.25, act=nn.ReLU(True)) if cspsppf else partial(SPPF, acts=[nn.SiLU(), nn.SiLU()])
        self.b0 = RepConv(c1=3,          c2=int(64*w),  s=2, act=F.relu)
        self.b1 = RepConv(c1=int(64*w),  c2=int(128*w), s=2, act=F.relu)
        self.b2 = BepC3(  c1=int(128*w), c2=int(128*w), e=csp_e, n=round(6*d))
        self.b3 = RepConv(c1=int(128*w), c2=int(256*w), s=2, act=F.relu)
        self.b4 = BepC3(  c1=int(256*w), c2=int(256*w), e=csp_e, n=round(12*d))
        self.b5 = RepConv(c1=int(256*w), c2=int(512*w), s=2, act=F.relu)
        self.b6 = BepC3(  c1=int(512*w), c2=int(512*w), e=csp_e, n=round(18*d))
        self.b7 = RepConv(c1=int(512*w), c2=int(1024*w),s=2, act=F.relu)
        self.b8 = BepC3(  c1=int(1024*w),c2=int(1024*w),e=csp_e, n=round(6*d))
        self.b9 = sppf(   c1=int(1024*w),c2=int(1024*w))

    def forward(self, x):
        x4  = self.b2(self.b1(self.b0(x)))      # p2/4
        x8  = self.b4(self.b3(x4))              # p3/8
        x16 = self.b6(self.b5(x8))              # p4/16
        x32 = self.b9(self.b8(self.b7(x16)))    # p5/32
        return x4, x8, x16, x32
    
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
        self.b0 = Conv(c, 256, 1, act=actV3)
        self.b1 = Conv(256,  128, 1, act=actV3)
    def forward(self, x16, x32):
        p32 = self.b0(x32)
        p16 = torch.cat([F.interpolate(self.b1(p32), scale_factor=2), x16], 1)
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
        x16 = torch.cat((self.c12(p16), F.interpolate(self.c14(p32), scale_factor=2)), 1)
        x17 = self.c17(x16)
        x20 = torch.cat((self.c11(p8), F.interpolate(self.c18(x17), scale_factor=2)), 1)
        xxx = self.cxx(x20)
        x23 = self.c23(torch.cat((self.c21(xxx), x17), 1))
        x26 = self.c26(torch.cat((self.c24(x23), p32), 1))
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
        x14 = self.n2(self.n1(torch.cat([self.up(x10),x6], 1)))                                      
        x17 = self.n3(torch.cat([self.up(x14),x4],  1))  # 17 (P3/8-small)
        x20 = self.n5(torch.cat([self.n4(x17),x14], 1))
        x23 = self.n7(torch.cat([self.n6(x20),x10], 1))
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
        self.n1 = C2fCIB(c1=int(512*w*(1+r)),   c2=int(512*w),   n=round(3*d), shortcut=True, cib=variant in "blx")
        self.n2 = C2f(c1=int(768*w),            c2=int(256*w),   n=round(3*d))
        self.n3 = Conv(c1=int(256*w),           c2=int(256*w),   k=3, s=2)
        self.n4 = C2fCIB(c1=int(768*w),         c2=int(512*w),   n=round(3*d), shortcut=True, cib=variant in "mblx")
        self.n5 = SCDown(c1=int(512*w),         c2=int(512*w),   k=3, s=2)
        self.n6 = C2fCIB(c1=int(512*w*(1+r)),   c2=int(512*w*r), n=round(3*d), shortcut=True, cib=True, lk=variant in "ns")

    def forward(self, x4, x6, x10):
        x13 = self.n1(torch.cat([self.up(x10),x6], 1))  # 13
        x16 = self.n2(torch.cat([self.up(x13),x4], 1))  # 16 (P3/8-small)
        x19 = self.n4(torch.cat([self.n3(x16),x13], 1)) # 19 (P4/16-medium)
        x22 = self.n6(torch.cat([self.n5(x19),x10], 1)) # 22 (P5/32-large)
        return [x16, x19, x22]

class HeadV11(nn.Module):
    def __init__(self, w, r, d, variant, is26=False):
        super().__init__()
        c3k = True if is26 else variant in "mlx"
        n   = 1    if is26 else 2
        self.up = nn.Upsample(scale_factor=2)
        self.n1 = C3k2(c1=int(512*w*(1+r)), c2=int(512*w),   n=round(2*d), c3k=c3k)
        self.n2 = C3k2(c1=int(512*w*2),     c2=int(256*w),   n=round(2*d), c3k=c3k)
        self.n3 = Conv(c1=int(256*w),       c2=int(256*w),   k=3, s=2)
        self.n4 = C3k2(c1=int(768*w),       c2=int(512*w),   n=round(2*d), c3k=c3k)
        self.n5 = Conv(c1=int(512*w),       c2=int(512*w),   k=3, s=2)
        self.n6 = C3k2(c1=int(512*w*(1+r)), c2=int(512*w*r), n=max(1,round(n*d)), c3k=True, attn=is26)

    def forward(self, x4, x6, x10):
        x13 = self.n1(torch.cat([self.up(x10),x6], 1))  # 13
        x16 = self.n2(torch.cat([self.up(x13),x4], 1))  # 16 (P3/8-small)
        x19 = self.n4(torch.cat([self.n3(x16),x13], 1)) # 19 (P4/16-medium)
        x22 = self.n6(torch.cat([self.n5(x19),x10], 1)) # 22 (P5/32-large)
        return [x16, x19, x22]

class HeadV12(nn.Module):
    def __init__(self, w, r, d):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.n1 = A2C2f(c1=int(512*w*(1+r)), c2=int(512*w), n=round(2*d), a2=False)
        self.n2 = A2C2f(c1=int(512*w*2),     c2=int(256*w), n=round(2*d), a2=False)
        self.n3 = Conv(c1=int(256*w),        c2=int(256*w), k=3, s=2)
        self.n4 = A2C2f(c1=int(768*w),       c2=int(512*w), n=round(2*d), a2=False)
        self.n5 = Conv(c1=int(512*w),        c2=int(512*w), k=3, s=2)
        self.n6 = C3k2(c1=int(512*w*(1+r)),  c2=int(512*w*r), n=round(2*d), c3k=True)
        
    def forward(self, x4, x6, x8):
        x11 = self.n1(torch.cat([self.up(x8),x6],   1)) # 11
        x14 = self.n2(torch.cat([self.up(x11),x4],  1)) # 14 (P3/8-small)
        x17 = self.n4(torch.cat([self.n3(x14),x11], 1)) # x17 (P4/16-medium)
        x20 = self.n6(torch.cat([self.n5(x17),x8],  1)) # 20 (P5/32-large)
        return [x14, x17, x20]
    
class BiFusion(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1[0], c2, 1, 1, act=nn.ReLU(True))
        self.cv2 = Conv(c1[1], c2, 1, 1, act=nn.ReLU(True))
        self.cv3 = Conv(c2*3,  c2, 1, 1, act=nn.ReLU(True))
        self.up  = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2, bias=True)
        self.do  = Conv(c2, c2, k=3, s=2, act=nn.ReLU(True))

    def forward(self, x):
        return self.cv3(torch.cat(( self.up(x[0]), self.cv1(x[1]), self.do(self.cv2(x[2]))), dim=1))
    
class RepBiFPANNeck(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        self.b0 = Conv(int(1024*w), int(256*w), k=1, s=1, act=nn.ReLU(True))
        self.b1 = BiFusion([int(512*w), int(256*w)], int(256*w))
        self.b2 = RepBlock(int(256*w), int(256*w), n=round(12*d))
        self.b3 = Conv(int(256*w), int(128*w), k=1, s=1, act=nn.ReLU(True))
        self.b4 = BiFusion([int(256*w), int(128*w)], int(128*w))
        self.b5 = RepBlock(int(128*w), int(128*w), n=round(12*d))
        self.b6 = Conv(int(128*w), int(128*w), k=3, s=2, act=nn.ReLU(True))
        self.b7 = RepBlock(int(128*w)+int(128*w), int(256*w), n=round(12*d))
        self.b8 = Conv(int(256*w), int(256*w), k=3, s=2, act=nn.ReLU(True))
        self.b9 = RepBlock(int(256*w)+int(256*w), int(512*w), n=round(12*d))

    def forward(self, x4, x8, x16, x32):
        a   = self.b0(x32)
        b   = self.b3(self.b2(self.b1([a, x16, x8])))
        p3  = self.b5(self.b4([b, x8, x4]))             # (P3/8-small)
        p4  = self.b7(torch.cat([self.b6(p3), b], 1))   # (P4/16-medium)
        p5  = self.b9(torch.cat([self.b8(p4), a], 1))   # (P5/32-large)
        return p3, p4, p5

class CSPRepBiFPANNeck(nn.Module):
    def __init__(self, w, d, csp_e=1/2):
        super().__init__()
        self.b0 = Conv(int(1024*w), int(256*w), k=1, s=1, act=nn.ReLU(True))
        self.b1 = BiFusion([int(512*w), int(256*w)], int(256*w))
        self.b2 = BepC3(int(256*w), int(256*w), n=round(12*d), e=csp_e)
        self.b3 = Conv(int(256*w), int(128*w), k=1, s=1, act=nn.ReLU(True))
        self.b4 = BiFusion([int(256*w), int(128*w)], int(128*w))
        self.b5 = BepC3(int(128*w), int(128*w), n=round(12*d), e=csp_e)
        self.b6 = Conv(int(128*w), int(128*w), k=3, s=2, act=nn.ReLU(True))
        self.b7 = BepC3(int(128*w)+int(128*w), int(256*w), n=round(12*d), e=csp_e)
        self.b8 = Conv(int(256*w), int(256*w), k=3, s=2, act=nn.ReLU(True))
        self.b9 = BepC3(int(256*w)+int(256*w), int(512*w), n=round(12*d), e=csp_e)

    def forward(self, x4, x8, x16, x32):
        a   = self.b0(x32)
        b   = self.b3(self.b2(self.b1([a, x16, x8])))
        p3  = self.b5(self.b4([b, x8, x4]))             # (P3/8-small)
        p4  = self.b7(torch.cat([self.b6(p3), b], 1))   # (P4/16-medium)
        p5  = self.b9(torch.cat([self.b8(p4), a], 1))   # (P5/32-large)
        return p3, p4, p5
    
def dist2box(dist, sxy, strides):
    lt, rb  = dist.chunk(2, 2)
    x1y1    = sxy - lt*strides
    x2y2    = sxy + rb*strides
    box     = torch.cat([x1y1,x2y2],-1)
    return box

@torch.no_grad()
def make_anchors(feats, strides): # anchor-free
    xys, strides2 = [], []
    for x, stride in zip(feats, strides):
        h, w    = x.shape[2], x.shape[3]
        sx      = (torch.arange(end=w, device=x.device) + 0.5) * stride
        sy      = (torch.arange(end=h, device=x.device) + 0.5) * stride
        sy, sx  = torch.meshgrid(sy, sx, indexing='ij')
        xy      = rearrange([sx,sy], 'c h w -> (h w) c')
        xys.append(xy)
        strides2.append(torch.full((h*w,1), fill_value=stride, device=x.device))
    return torch.cat(xys,0), torch.cat(strides2,0)

@torch.no_grad()
def make_anchors_ab(feats, strides, scales, anchors): # anchor-based
    xys, awhs, strides2, scales2 = [], [], [], []
    for x, awh , stride, scale in zip(feats, anchors, strides, scales):
        a, h, w = awh.shape[0], x.shape[2], x.shape[3]
        sx      = (torch.arange(end=w, device=x.device) + 0.5) * stride
        sy      = (torch.arange(end=h, device=x.device) + 0.5) * stride
        sy, sx  = torch.meshgrid(sy, sx, indexing='ij')
        xy      = repeat([sx,sy], 'c h w -> (a h w) c', a=a)
        awh     = repeat(awh, 'a c -> (a h w) c', h=h, w=w)
        xys.append(xy)
        awhs.append(awh)
        strides2.append(torch.full((a*h*w,1), fill_value=stride, device=x.device))
        scales2.append(torch.full((a*h*w,1), fill_value=scale, device=x.device))
    return *pack(xys, '* c'), torch.cat(awhs,0), torch.cat(strides2,0), torch.cat(scales2,0)

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

    def to_wh(self, wh, awh):
        match self.v7:
            case True:  return awh * (wh.sigmoid() ** 2) * 4
            case False: return awh * wh.exp()
    
    def forward(self, xs, targets=None):
        sxy, ps, awh, strides, scales = make_anchors_ab(xs, self.strides, self.scales, self.anchors_wh)
        feats           = [rearrange(n(x), 'b (a f) h w -> b (a h w) f', f=self.nc+5) for x,n in zip(xs, self.cv)]
        xy, wh, l, cls  = torch.cat(feats,1).split((2,2,1,self.nc), -1) 
        xy              = strides * ((xy.sigmoid() - 0.5) * scales) + sxy
        wh              = self.to_wh(wh, awh)
        box             = torch.cat([xy-wh/2, xy+wh/2], -1)
        pred            = torch.cat([box, l.sigmoid(), cls.sigmoid()], -1)

        if exists(targets):
            anchors               = torch.cat([sxy-awh/2, sxy+awh/2],-1)
            tboxes, tscores, tcls = assigner.atss(anchors, targets, [p[0] for p in ps], self.nc, 9)
            # tboxes, tscores, tcls = assigner.tal(box, cls.sigmoid(), sxy, targets, 9, 0.5, 6.0)
            mask                  = tscores > 0

            # CIOU loss (positive samples)
            tgt_scores_sum = tscores.sum().clamp(min=1.0)
            weight   = tscores[mask]
            loss_iou = (torchvision.ops.complete_box_iou_loss(box[mask], tboxes[mask], reduction='none') * weight).sum() / tgt_scores_sum
            
            # Class loss (positive samples)
            loss_cls = (F.binary_cross_entropy_with_logits(cls[mask], tcls[mask], reduction='none') * weight.unsqueeze(-1)).sum() / tgt_scores_sum
            
            # Objectness loss (positive + negative samples)
            loss_obj = F.binary_cross_entropy_with_logits(l.squeeze(-1), tscores, reduction='sum') / tgt_scores_sum

        return pred if not exists(targets) else (pred, {'iou': loss_iou, 'cls': loss_cls, 'obj': loss_obj})

class Detect(nn.Module):
    def __init__(self, nc=80, ch=(), dfl=True, separable=False, end2end=False):
        super().__init__()
        def spconv(c1, c2, k): return nn.Sequential(Conv(c1,c1,k,g=c1),Conv(c1,c2,1))
        conv = spconv if separable else Conv
        self.nc         = nc
        self.dfl        = dfl
        self.reg_max    = 16 if dfl else 1
        self.strides    = [8, 16, 32]
        c2              = max((16, ch[0] // 4, self.reg_max * 4))
        c3              = max(ch[0], min(nc, 100))  # channels
        self.cv2        = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3        = nn.ModuleList(nn.Sequential(conv(x, c3, 3), conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.r          = nn.Parameter(torch.arange(self.reg_max).float(), requires_grad=False) if dfl else None
        if end2end: self.one2one_cv2 = deepcopy(self.cv2)
        if end2end: self.one2one_cv3 = deepcopy(self.cv3)

    def forward_private(self, xs, cv2, cv3, targets=None):
        sxy, strides    = make_anchors(xs, self.strides)
        feats           = [rearrange(torch.cat((c1(x), c2(x)), 1), 'b f h w -> b (h w) f') for x,c1,c2 in zip(xs, cv2, cv3)]
        dist, logits    = torch.cat(feats, 1).split((4 * self.reg_max, self.nc), -1)
        dist            = rearrange(dist, 'b n (k r) -> b n k r', k=4)
        ltrb            = torch.einsum('bnkr, r -> bnk', dist.softmax(-1), self.r) if self.dfl else dist.squeeze(-1)
        box             = dist2box(ltrb, sxy, strides)
        probs           = logits.sigmoid()
        pred            = torch.cat((box, probs), -1)

        if exists(targets):
            tboxes, tscores, tcls = assigner.tal(box, probs, sxy, targets, 9, 0.5, 6.0)
            mask                  = tscores > 0

            # CIOU loss (positive samples)
            tgt_scores_sum = tscores.sum().clamp(min=1.0)
            weight   = tscores[mask]
            loss_iou = (torchvision.ops.complete_box_iou_loss(box[mask], tboxes[mask], reduction='none') * weight).sum() / tgt_scores_sum

            # DFL loss (positive samples)
            loss_dfl = dfl_loss(tboxes, mask, tgt_scores_sum, sxy, strides, dist) if self.dfl else torch.zeros((), device=box.device)
            
            # Class loss (positive samples + negative)
            loss_cls = F.binary_cross_entropy_with_logits(logits, tcls*tscores.unsqueeze(-1), reduction='sum') / tgt_scores_sum

        return pred if not exists(targets) else (pred, {'iou': loss_iou, 'dfl': loss_dfl, 'cls': loss_cls})
        
    def forward(self, xs, targets=None):
        return self.forward_private(xs, self.cv2, self.cv3, targets)
    
class DetectV6(nn.Module):
    def __init__(self, nc=80, ch=(), use_dfl=False, distill=False):
        super().__init__()
        self.nc         = nc                        # number of classes
        self.na         = 1                         # number of anchors
        self.reg_max    = 16 if use_dfl else 0      # DFL channels
        self.strides    = [8, 16, 32]               # strides
        # Decoupled head
        self.stems          = nn.ModuleList([Conv(c1=c, c2=c, k=1, act=nn.SiLU(True)) for c in ch])
        self.cls_convs      = nn.ModuleList([Conv(c1=c, c2=c, k=3, act=nn.SiLU(True)) for c in ch])
        self.reg_convs      = nn.ModuleList([Conv(c1=c, c2=c, k=3, act=nn.SiLU(True)) for c in ch])
        self.cls_preds      = nn.ModuleList([nn.Conv2d(c, self.na*self.nc,            kernel_size=1) for c in ch])
        self.reg_preds_dist = nn.ModuleList([nn.Conv2d(c, 4*(self.na + self.reg_max), kernel_size=1) for c in ch])
        self.reg_preds      = nn.ModuleList([nn.Conv2d(c, 4*(self.na),                kernel_size=1) for c in ch]) if distill else None
        self.r              = nn.Parameter(torch.arange(self.reg_max+1).float(), requires_grad=False) if use_dfl else None
    
    def forward(self, xs, targets=None):
        sxy, strides = make_anchors(xs, self.strides)
        xs          = [l(x) for l,x in zip(self.stems, xs)]
        cls         = torch.cat([rearrange(c2(c1(x)), 'b f h w -> b (h w) f') for c1,c2,x in zip(self.cls_convs, self.cls_preds, xs)], 1)
        reg         = [c1(x) for c1,x in zip(self.reg_convs, xs)]
        dist        = torch.cat([rearrange(c2(x), 'b (k r) h w -> b (h w) k r', k=4) for c2,x in zip(self.reg_preds_dist, reg)], 1)
        ltrb        = torch.einsum('bnkr, r -> bnk', dist.softmax(-1), self.r)
        box         = dist2box(ltrb, sxy, strides)
        ltrb        = torch.cat([rearrange(c2(x), 'b k h w -> b (h w) k') for c2,x in zip(self.reg_preds, reg)], 1) if exists(self.reg_preds) else None
        box_distill = dist2box(ltrb, sxy, strides) if exists(ltrb) else None
        pred        = torch.cat((box, cls.sigmoid()), -1)

        if exists(targets):
            # awh                     = torch.full_like(sxy, fill_value=5.0) * strides # Fake height and width for the sake of ATSS
            # anchors                 = torch.cat([sxy-awh/2, sxy+awh/2],-1)
            # tboxes, tscores, tcls   = assigner.atss(anchors, targets, [p[0] for p in ps], self.nc, 9)
            tboxes, tscores, tcls   = assigner.tal(box, cls.sigmoid(), sxy, targets, 9, 0.5, 6.0)
            # tboxes, tscores, tcls   = assigner.fcos(sxy, targets, self.nc)
            mask                    = tscores > 0

            # CIOU loss (positive samples)
            tgt_scores_sum = max(tscores.sum(), 1)
            weight           = tscores[mask]
            loss_iou         = (torchvision.ops.complete_box_iou_loss(box[mask],         tboxes[mask], reduction='none') * weight).sum() / tgt_scores_sum
            loss_iou_distill = (torchvision.ops.complete_box_iou_loss(box_distill[mask], tboxes[mask], reduction='none') * weight).sum() / tgt_scores_sum if exists(box_distill) else 0
            
            # DFL loss (positive samples)
            loss_dfl = dfl_loss(tboxes, mask, tgt_scores_sum, sxy, strides, dist)
            
            # Class loss (positive samples + negative)
            loss_cls = F.binary_cross_entropy_with_logits(cls, tcls*tscores.unsqueeze(-1), reduction='sum') / tgt_scores_sum

        return pred if not exists(targets) else (pred, {'iou': loss_iou+loss_iou_distill, 'dfl': loss_dfl, 'cls': loss_cls})
    
class YoloBase(nn.Module):
    def __init__(self, net, fpn, head, variant=None):
        super().__init__()
        self.v      = variant
        self.net    = net
        self.fpn    = fpn
        self.head   = head
    
    def forward(self, x, targets=None):
        x = self.net(x)
        x = self.fpn(*x)
        return self.head(x, targets=targets)
    
class Yolov3(YoloBase):
    def __init__(self, nclasses, spp):
        super().__init__(Darknet53(),
                         HeadV3(spp),
                         DetectV3(nclasses, [8,16,32], ANCHORS_V3, [1,1,1], ch=[(128, 256), (256, 512), (512, 1024)]))
    
class Yolov3Tiny(YoloBase):
    def __init__(self, nclasses):
        super().__init__(BackboneV3Tiny(),
                         HeadV3Tiny(1024),
                         DetectV3(nclasses, [16,32], ANCHORS_V3_TINY, [1,1], ch=[(384, 256), (256, 512)]))
    
class Yolov4(YoloBase):
    def __init__(self, nclasses, act=actV4):
        super().__init__(BackboneV4(act),
                         HeadV4(actV3),
                         DetectV3(nclasses, [8,16,32], ANCHORS_V4, [1.2, 1.1, 1.05], ch=[(128, 256), (256, 512), (512, 1024)]))

class Yolov4Tiny(YoloBase):
    def __init__(self, nclasses):
        super().__init__(BackboneV4Tiny(),
                         HeadV3Tiny(512),
                         DetectV3(nclasses, [16,32], ANCHORS_V3_TINY, [1.05,1.5], ch=[(384, 256), (256, 512)]))

class Yolov7(YoloBase):
    def __init__(self, nclasses):
        super().__init__(BackboneV7(),
                         HeadV7(),
                         DetectV3(nclasses, [8,16,32], ANCHORS_V4, [2,2,2], ch=[(128,256), (256,512), (512,1024)], is_v7=True))

class Yolov5(YoloBase):
    def __init__(self, variant, num_classes):
        d, w, r = get_variant_multiplesV5(variant)
        super().__init__(BackboneV5(w, r, d),
                         HeadV5(w, r, d),
                         Detect(num_classes, ch=(int(256*w), int(512*w), int(512*w*2)), separable=False, dfl=True, end2end=False),
                         variant)
        
class Yolov8(YoloBase):
    def __init__(self, variant, num_classes):
        d, w, r = get_variant_multiplesV8(variant)
        super().__init__(BackboneV8(w, r, d),
                         HeadV8(w, r, d),
                         Detect(num_classes, ch=(int(256*w), int(512*w), int(512*w*r)), separable=False, dfl=True, end2end=False),
                         variant)

class Yolov10(YoloBase):
    def __init__(self, variant, num_classes):
        d, w, r = get_variant_multiplesV10(variant)
        super().__init__(BackboneV10(w, r, d, variant),
                         HeadV10(w, r, d, variant),
                         Detect(num_classes, ch=(int(256*w), int(512*w), int(512*w*r)), separable=True, dfl=True, end2end=True),
                         variant)

class Yolov11(YoloBase):
    def __init__(self, variant, num_classes):
        d, w, r = get_variant_multiplesV11(variant)
        super().__init__(BackboneV11(w, r, d, variant),
                         HeadV11(w, r, d, variant),
                         Detect(num_classes, ch=(int(256*w), int(512*w), int(512*w*r)), separable=True, dfl=True, end2end=False),
                         variant)

class Yolov26(YoloBase):
    def __init__(self, variant, num_classes):
        d, w, r = get_variant_multiplesV26(variant)
        super().__init__(BackboneV11(w, r, d, variant, sppf_shortcut=True, sppf_acts=[nn.Identity(), nn.SiLU(True)]),
                         HeadV11(w, r, d, variant, is26=True),
                         Detect(num_classes, ch=(int(256*w), int(512*w), int(512*w*r)), separable=True, dfl=False, end2end=True),
                         variant)
        
class Yolov12(YoloBase):
    def __init__(self, variant, num_classes):
        d, w, r = get_variant_multiplesV12(variant)
        super().__init__(BackboneV12(w, r, d, variant),
                         HeadV12(w, r, d),
                         Detect(num_classes, ch=(int(256*w), int(512*w), int(512*w*r)), separable=True),
                         variant)

class Yolov6(YoloBase):
    def __init__(self, variant, num_classes):
        d, w, csp, csp_e, distill = get_variant_multiplesV6(variant)
        super().__init__(CSPBepBackbone(w, d, csp_e=csp_e)   if csp else EfficientRep(w, d, cspsppf=True),
                         CSPRepBiFPANNeck(w, d, csp_e=csp_e) if csp else RepBiFPANNeck(w, d),
                         DetectV6(num_classes, ch=(int(128*w), int(256*w), int(512*w)), use_dfl=True, distill=distill),
                         variant)

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