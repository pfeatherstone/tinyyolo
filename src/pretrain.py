import argparse
from   PIL import ImageFile
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms.v2 as v2
import lightning.pytorch as pl
from   lightning.pytorch.loggers import TensorBoardLogger
from   lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import matplotlib.pyplot as plt
from   models import *
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument("--nepochs",      type=int,   default=100,    help="Number of epochs")
parser.add_argument("--batchsize",    type=int,   default=32,     help="Batch size")
parser.add_argument("--nwarmup",      type=int,   default=1000,   help="number of warmup steps")
parser.add_argument("--lr",           type=float, default=0.001,  help="Initial learning rate")
parser.add_argument("--nworkers",     type=int,   default=0,      help="Number of data workers. If 0, set to mp.cpu_count()/2")
args = parser.parse_args()
args.nworkers = torch.multiprocessing.cpu_count() // 2 if args.nworkers == 0 else args.nworkers

class LitModule(pl.LightningModule):
    def __init__(self, net, nsteps):
        super().__init__()
        self.net        = net
        self.nsteps     = nsteps
        self.augment    = v2.Compose([
            v2.RandomPhotometricDistort(p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.5),
            v2.RandomSolarize(threshold=192.0/255.0, p=0.5),
            v2.RandomHorizontalFlip(p=0.5)])

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.trainer.num_training_batches, is_training=True)

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, self.trainer.num_val_batches[0], is_training=False)

    def step(self, imgs, batch_idx, nbatches, is_training):
        z1 = self.net(self.augment(imgs))
        z2 = self.net(self.augment(imgs))
        loss, cross = barlow_loss(z1, z2, 5e-3)

        label = "train" if is_training else "val"
        self.log("loss/" + label, loss.item(), on_step=True, on_epoch=not is_training)

        if self.trainer.is_global_zero:
            summary     = self.logger.experiment
            epoch       = self.current_epoch
            totalBatch  = (epoch + batch_idx / nbatches) * 1000

            if batch_idx % 50 == 0:
                summary.add_image("C/" + label, cross, global_step=totalBatch, dataformats='HW')

        self.trainer.strategy.barrier()
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=args.lr, 
                                                        total_steps=self.nsteps,
                                                        pct_start=args.nwarmup/self.nsteps)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': "step", "frequency": 1}}

def Collator(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0)

transform = v2.Compose([
    v2.RandomPhotometricDistort(p=0.8),
    v2.RandomGrayscale(p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.5),
    v2.RandomSolarize(threshold=192.0/255.0, p=0.5),
    v2.RandomHorizontalFlip(p=0.5)])

dataset          = torchvision.datasets.Caltech256('/home/pf/Downloads/caltech256', transform=v2.Compose([v2.ToImage(), v2.GrayscaleToRgb(), v2.Resize((640,640), antialias=True), v2.ToDtype(torch.float32, scale=True)]), download=True)
trainset, valset = torch.utils.data.random_split(dataset, [0.8, 0.2], torch.Generator())
trainset         = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, collate_fn=Collator, num_workers=args.nworkers, shuffle=True)
valset           = torch.utils.data.DataLoader(valset,   batch_size=args.batchsize, collate_fn=Collator, num_workers=args.nworkers)
nsteps           = len(trainset) * args.nepochs

for img, targets in valset:
    img2 = transform(img)
    plt.figure()
    plt.imshow((img.permute(1,2,0)*255).to(torch.uint8).numpy())
    plt.figure()
    plt.imshow((img2.permute(1,2,0)*255).to(torch.uint8).numpy())
    plt.show()

d, w, r = get_variant_multiplesV8('n')
net = BackboneV8(w, r, d)
net = BarlowTwinsHead(net, int(512*w*r), 2048, 1024)
net = LitModule(net, nsteps)

trainer = pl.Trainer(max_epochs=args.nepochs,
                     accelerator='gpu',
                     num_sanity_val_steps=0,
                     logger=TensorBoardLogger(save_dir="../runs", flush_secs=10),
                     callbacks= [LearningRateMonitor(logging_interval='step', log_momentum=True),
                                 ModelCheckpoint(filename='epoch_{epoch}-loss_{loss/val_epoch}',
                                                 monitor='loss/sum/val_epoch', 
                                                 auto_insert_metric_name=False)])

trainer.fit(model=net, train_dataloaders=trainset, val_dataloaders=valset)