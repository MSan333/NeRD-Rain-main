"""
exp4: SFStar 模块训练（单卡续训版）
- 从 SFStar_2GPU_lr2e4/model_best.pth (epoch 8) 恢复
- 单卡 GPU 3, lr=2e-4 → 1e-6, warmup 3 epoch, 500 epoch
- 续训：恢复模型权重，重新初始化 optimizer（lr 从 schedule 对应位置开始）
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from model import MultiscaleNet as myNet
from losses import CharbonnierLoss, EdgeLoss, fftLoss, HierarchicalAdaptiveFreqLoss
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from get_parameter_number import get_parameter_number
import kornia
from torch.utils.tensorboard import SummaryWriter
import argparse
import swanlab

from skimage import img_as_ubyte

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

parser = argparse.ArgumentParser(description='Image Deraininig - exp4 SFStar (Single GPU)')

parser.add_argument('--train_dir', default='../data/Rain200L/train/', type=str)
parser.add_argument('--val_dir', default='../data/Rain200L/test/', type=str)
parser.add_argument('--model_save_dir', default='./ckpt/', type=str)
parser.add_argument('--resume_weights', default='./ckpt/Deraininig/models/SFStar_2GPU_lr2e4/model_best.pth', type=str)
parser.add_argument('--mode', default='Deraininig', type=str)
parser.add_argument('--session', default='SFStar_1GPU_lr2e4', type=str)
parser.add_argument('--patch_size', default=256, type=int)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--val_epochs', default=1, type=int)
parser.add_argument('--hafl_warmup_epochs', default=50, type=int)
args = parser.parse_args()

######### SwanLab Init ###########
swanlab.login(api_key="o4MGQAOSX8rGztH69Jj5P")
swanlab.init(
    project="NeRD-Rain",
    experiment_name="SFStar_1GPU_lr2e4",
    config={
        **vars(args),
        "start_lr": 2e-4,
        "end_lr": 1e-6,
        "warmup_epochs": 3,
        "hafl_warmup_epochs": args.hafl_warmup_epochs,
        "optimizer": "Adam",
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "gpu": "single GPU 3",
        "model_variant": "SFStar",
        "description": "exp4: SFStar单卡续训，从epoch8恢复",
    },
)

mode = args.mode
session = args.session
patch_size = args.patch_size

model_dir = os.path.join(args.model_save_dir, mode, 'models', session)
utils.mkdir(model_dir)

train_dir = args.train_dir
val_dir = args.val_dir

num_epochs = args.num_epochs
batch_size = args.batch_size
val_epochs = args.val_epochs

start_lr = 2e-4
end_lr = 1e-6

######### Model ###########
model_restoration = myNet()
get_parameter_number(model_restoration)
model_restoration = model_restoration.cuda()

######### Resume from checkpoint ###########
start_epoch = 1
best_psnr = 0
best_epoch = 0

if args.resume_weights and os.path.exists(args.resume_weights):
    checkpoint = torch.load(args.resume_weights, map_location='cuda:0')
    model_restoration.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"==> Resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")
    print(f"==> Loaded weights from: {args.resume_weights}")
else:
    print("==> No checkpoint found, training from scratch")

######### Optimizer & Scheduler ###########
optimizer = optim.Adam(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - warmup_epochs, eta_min=end_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

# Step scheduler to correct position for resumed epoch
for i in range(1, start_epoch):
    scheduler.step()

current_lr = scheduler.get_lr()[0]
print('------------------------------------------------------------------------------')
print(f"==> exp4 Single GPU: SFStar model, resume training from epoch {start_epoch}")
print(f"==> LR: {current_lr:.6f} (schedule: 2e-4 → 1e-6, CosineAnnealing)")
print(f"==> Total epochs: {num_epochs}, remaining: {num_epochs - start_epoch + 1}")
print('------------------------------------------------------------------------------')

######### Loss ###########
criterion_char = CharbonnierLoss()
criterion_edge = EdgeLoss()
criterion_fft = fftLoss()
criterion_L1 = nn.L1Loss(size_average=True)
criterion_hafl = HierarchicalAdaptiveFreqLoss()

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size': patch_size})
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size': patch_size})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, num_epochs + 1))
print('===> Loading datasets')
print(f'===> Train samples: {len(train_dataset)}, iterations/epoch: {len(train_loader)}')

writer = SummaryWriter(model_dir)
iter_count = 0

for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_fft_loss = 0
    epoch_char_loss = 0
    epoch_edge_loss = 0
    epoch_l1_loss = 0
    epoch_hafl_loss = 0

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target_ = data[0].cuda()
        input_ = data[1].cuda()
        target = kornia.geometry.transform.build_pyramid(target_, 3)
        restored = model_restoration(input_)

        loss_fft = criterion_fft(restored[0], target[0]) + criterion_fft(restored[1], target[1]) + criterion_fft(restored[2], target[2])
        loss_char = criterion_char(restored[0], target[0]) + criterion_char(restored[1], target[1]) + criterion_char(restored[2], target[2])
        loss_edge = criterion_edge(restored[0], target[0]) + criterion_edge(restored[1], target[1]) + criterion_edge(restored[2], target[2])
        loss_l1 = criterion_L1(restored[3], target[1]) + criterion_L1(restored[5], target[2])
        loss_hafl = criterion_hafl(restored[0], target[0], epoch, num_epochs, args.hafl_warmup_epochs)
        loss = loss_char + 0.01 * loss_fft + 0.05 * loss_edge + 0.1 * loss_l1 + 0.1 * loss_hafl
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_fft_loss += loss_fft.item()
        epoch_char_loss += loss_char.item()
        epoch_edge_loss += loss_edge.item()
        epoch_l1_loss += loss_l1.item()
        epoch_hafl_loss += loss_hafl.item()
        iter_count += 1
        writer.add_scalar('loss/fft_loss', loss_fft, iter_count)
        writer.add_scalar('loss/char_loss', loss_char, iter_count)
        writer.add_scalar('loss/edge_loss', loss_edge, iter_count)
        writer.add_scalar('loss/l1_loss', loss_l1, iter_count)
        writer.add_scalar('loss/hafl_loss', loss_hafl, iter_count)
        writer.add_scalar('loss/iter_loss', loss, iter_count)

    swanlab.log({
        "loss/epoch_loss": epoch_loss,
        "loss/fft_loss": epoch_fft_loss,
        "loss/char_loss": epoch_char_loss,
        "loss/edge_loss": epoch_edge_loss,
        "loss/l1_loss": epoch_l1_loss,
        "loss/hafl_loss": epoch_hafl_loss,
        "epoch": epoch,
    }, step=epoch)
    writer.add_scalar('loss/epoch_loss', epoch_loss, epoch)

    #### Evaluation ####
    if epoch % val_epochs == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            with torch.no_grad():
                restored = model_restoration(input_)

            for res, tar in zip(restored[0], target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        writer.add_scalar('val/psnr', psnr_val_rgb, epoch)
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best.pth"))

        swanlab.log({"val/psnr": psnr_val_rgb, "val/best_psnr": best_psnr, "epoch": epoch}, step=epoch)
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

    scheduler.step()

    current_lr = scheduler.get_lr()[0]
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, current_lr))
    print("------------------------------------------------------------------")
    swanlab.log({"lr": current_lr, "epoch_time": time.time() - epoch_start_time, "epoch": epoch}, step=epoch)

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

writer.close()
swanlab.finish()
