import os
import argparse
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from   torchvision.transforms import v2
import lightning.pytorch as pl
from   lightning.pytorch.loggers import TensorBoardLogger
from   lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import matplotlib.pyplot as plt
from   models import Yolov4Tiny, nms, COCO_NAMES

parser = argparse.ArgumentParser()
parser.add_argument("--nepochs",      type=int,   default=100,    help="Number of epochs")
parser.add_argument("--batchsize",    type=int,   default=32,     help="Batch size")
parser.add_argument("--nwarmup",      type=int,   default=1000,   help="number of warmup steps")
parser.add_argument("--lr",           type=float, default=0.001,  help="Initial learning rate")
parser.add_argument("--trainRoot",    type=str,   required=True,  help="Root folder of training directory")
parser.add_argument("--trainAnn",     type=str,   required=True,  help="Training annotations file")
parser.add_argument("--valRoot",      type=str,   required=True,  help="Root folder of validation directory")
parser.add_argument("--valAnn",       type=str,   required=True,  help="Validation annotations file")
parser.add_argument("--nworkers",     type=int,   default=0,      help="Number of data workers. If 0, set to mp.cpu_count()/2")
args = parser.parse_args()
args.nworkers = torch.multiprocessing.cpu_count() // 2 if args.nworkers == 0 else args.nworkers

class RandomResize(v2.Transform):
    def __init__(self, min_size: int, max_size: int, max_stride: int = 32):
        super().__init__()
        self.min_size   = min_size
        self.max_size   = max_size
        self.max_stride = max_stride

    def _get_params(self, flat_inputs):
        size = ((torch.randint(self.min_size, self.max_size, (2,)) + self.max_stride - 1) // self.max_stride) * self.max_stride
        return dict(H=size[0], W=size[1])

    def _transform(self, inpt, params):
        return self._call_kernel(v2.functional.resize, inpt, size=(params['H'], params['W']), interpolation=v2.InterpolationMode.BILINEAR, antialias=True)

class CocoWrapper(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=[]):
        super().__init__()
        transforms   = v2.Compose([v2.ToImage(), *transforms, v2.ToDtype(torch.float32, scale=True)])
        dataset      = torchvision.datasets.CocoDetection(root, annFile, transforms=transforms)
        cat_ids      = dataset.coco.getCatIds()
        cats         = dataset.coco.loadCats(cat_ids)
        self.names   = [cat["name"] for cat in cats]
        self.ids     = {cat: id for id, cat in enumerate(cat_ids)}
        self.dataset = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=["boxes", "labels"])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, target = self.dataset[index]
        classes     = torch.tensor([self.ids[i.item()] for i in target['labels']]).unsqueeze(-1) if 'labels' in target else torch.zeros(0,1)
        boxes       = target['boxes'] if 'boxes' in target else torch.zeros(0,4)
        target      = torch.cat([boxes,classes], -1)
        return img, target

def CocoCollator(batch):
    imgs, targets   = zip(*batch)
    N               = max(t.shape[0] for t in targets)
    targets         = [F.pad(t, (0,0,0,N-t.shape[0]), value=-1) for t in targets]
    H               = max(x.shape[1] for x in imgs)
    W               = max(x.shape[2] for x in imgs)
    imgs            = [F.pad(x, (0,W-x.shape[2],0,H-x.shape[1]), value=0) for x in imgs]
    imgs            = torch.stack(imgs, 0)
    targets         = torch.stack(targets, 0)
    return imgs, targets

def createOptimizer(self: torch.nn.Module, momentum=0.9, lr=0.001, decay=0.0001):
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params    = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params  = [p for n, p in param_dict.items() if p.dim() < 2]
    assert len(decay_params) + len(nodecay_params) == len(list(filter(lambda p: p.requires_grad, self.parameters()))), "bad split"
    optim_groups = [
        {'params': decay_params,   'weight_decay': decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params    = sum(p.numel() for p in decay_params)
    num_nodecay_params  = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors       : {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors   : {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(momentum, 0.999), fused=True)
    return optimizer

class LitModule(pl.LightningModule):
    def __init__(self, net, nsteps):
        super().__init__()
        self.net = net
        self.nsteps = nsteps

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.trainer.num_training_batches, is_training=True)

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, self.trainer.num_val_batches[0], is_training=False)
    
    def forward(self, x):
        # For ONNX export only
        return self.net(x)

    def step(self, batch, batch_idx, nbatches, is_training):
        imgs, targets = batch
        preds, losses = self.net(imgs, targets)
        loss          = 3.54 * losses['iou'] + 37.4 * losses['cls'] + 10 * losses['obj'] + 100.0 * losses['noobj']

        label = "train" if is_training else "val"
        self.log("loss/obj/"   + label, losses['obj'].item(),   logger=False, prog_bar=False, on_step=True)
        self.log("loss/noobj/" + label, losses['noobj'].item(), logger=False, prog_bar=False, on_step=True)
        self.log("loss/cls/"   + label, losses['cls'].item(),   logger=False, prog_bar=False, on_step=True)
        self.log("loss/iou/"   + label, losses['iou'].item(),   logger=False, prog_bar=False, on_step=True)
        self.log("loss/sum/"   + label, loss.item(),            logger=False, prog_bar=True, on_step=True, on_epoch=True)

        if self.trainer.is_global_zero:
            summary     = self.logger.experiment
            epoch       = self.current_epoch
            totalBatch  = (epoch + batch_idx / nbatches) * 1000

            summary.add_scalars("loss/obj",   {label: losses['obj'].item()},   totalBatch)
            summary.add_scalars("loss/noobj", {label: losses['noobj'].item()}, totalBatch)
            summary.add_scalars("loss/cls",   {label: losses['cls'].item()},   totalBatch)
            summary.add_scalars("loss/iou",   {label: losses['iou'].item()},   totalBatch)
            summary.add_scalars("loss/sum",   {label: loss.item()},            totalBatch)

            if batch_idx % 50 == 0:
                with torch.no_grad():
                    _, preds = nms(preds[0:1], 0.3, 0.5, True)
                    img      = (imgs[0]*255).to(torch.uint8)
                    canvas   = torchvision.utils.draw_bounding_boxes(img, preds[:,:4], [COCO_NAMES[i] for i in preds[:, 5:].argmax(-1).long()])
                    fig = plt.figure()
                    plt.imshow(canvas.permute(1,2,0).cpu())
                    summary.add_figure('preds/'+label, fig, totalBatch)
        self.trainer.strategy.barrier()
        return loss
    
    def configure_optimizers(self):
        optimizer = createOptimizer(self, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=args.lr, 
                                                        total_steps=self.nsteps,
                                                        pct_start=args.nwarmup/self.nsteps)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {'scheduler': scheduler, 'interval': "step", "frequency": 1}
        }
    
class OnnxCheckpoint(pl.callbacks.Checkpoint):
    def __init__(self, monitor):
        self.monitor    = monitor
        self.lastValue  = 100000
        self.lastOnnx   = ""

    def on_validation_end(self, trainer: pl.Trainer, lit: LitModule):
        ckpdir = trainer.log_dir + "/checkpoints/"
        epoch  = trainer.current_epoch

        if trainer.is_global_zero:
            if self.monitor in trainer.callback_metrics:
                newVal = trainer.callback_metrics[self.monitor].item()
                if newVal < self.lastValue:
                    self.lastValue = newVal
                    os.makedirs(ckpdir, exist_ok=True)
                    if os.path.exists(self.lastOnnx):
                        os.remove(self.lastOnnx)
                    self.lastOnnx = ckpdir + "epoch={}_{}={:.4f}.onnx".format(epoch, self.monitor.replace("/", "_"), newVal)
                    x = torch.randn(4,3,416,416)
                    lit.to_onnx(self.lastOnnx, input_sample=(x,), opset_version=17, 
                                input_names=['imgs'], 
                                output_names=['preds'], 
                                dynamic_axes={'imgs':  {0: 'B', 2: 'H', 3: 'W'},
                                              'preds': {0: 'B', 1: 'D'}})
                    
        trainer.strategy.barrier()

torch.set_float32_matmul_precision('medium')

transforms = [
    RandomResize(320,640,32),
    v2.RandomPhotometricDistort(p=0.7),
    v2.RandomPosterize(bits=4, p=0.7)
]
                                   
trainset    = CocoWrapper(args.trainRoot, args.trainAnn, transforms=transforms)
valset      = CocoWrapper(args.valRoot,   args.valAnn,   transforms=[v2.Resize((416,416), antialias=True)])
nclasses    = len(valset.names)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, collate_fn=CocoCollator, num_workers=args.nworkers)
valLoader   = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, collate_fn=CocoCollator, num_workers=args.nworkers)
nsteps      = len(trainLoader) * args.nepochs

net = Yolov4Tiny(nclasses)
net = LitModule(net, nsteps)

trainer = pl.Trainer(max_epochs=args.nepochs,
                     accelerator='gpu',
                     num_sanity_val_steps=0,
                     logger=TensorBoardLogger(save_dir="../runs", flush_secs=10),
                     callbacks= [LearningRateMonitor(logging_interval='step', log_momentum=True),
                                 OnnxCheckpoint(monitor='loss/sum/val_epoch'),
                                 ModelCheckpoint(filename='epoch_{epoch}-loss_{loss/sum/val_epoch}',
                                                 monitor='loss/sum/val_epoch', 
                                                 auto_insert_metric_name=False)])

trainer.fit(model=net, train_dataloaders=trainLoader, val_dataloaders=valLoader)