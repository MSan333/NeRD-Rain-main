"""
train_lr7e5.py — 基于 exp3_lr1e4 best checkpoint 的降 LR 续训版本

修改点：
  - session: DDP_2GPU_exp3_lr7e5（新目录保存）
  - start_lr: 1e-4 → 7e-5
  - end_lr:   2e-6 → 1e-6
  - warmup:   关闭（续训不需要 warmup）
  - RESUME:   True，从 exp3_lr1e4 的 model_best.pth 续训
  - num_epochs: 从 checkpoint epoch 起再跑 500 epoch
  - Resume 路径硬编码指向原 session DDP_2GPU_exp3_lr1e4 的 best checkpoint
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from model import MultiscaleNet as myNet
#from model_S import MultiscaleNet as myNet
from losses import CharbonnierLoss, EdgeLoss, fftLoss, HierarchicalAdaptiveFreqLoss
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

start_epoch = 1

parser = argparse.ArgumentParser(description='Image Deraininig')

parser.add_argument('--train_dir', default='../data/Rain200L/train/', type=str, help='Directory of train images')
parser.add_argument('--val_dir', default='../data/Rain200L/test/', type=str, help='Directory of validation images')
parser.add_argument('--model_save_dir', default='./ckpt/', type=str, help='Path to save weights')
parser.add_argument('--pretrain_weights', default='', type=str, help='Path to pretrain-weights')
parser.add_argument('--mode', default='Deraininig', type=str)
parser.add_argument('--session', default='DDP_2GPU_exp3_lr7e5', type=str, help='session')
parser.add_argument('--patch_size', default=256, type=int, help='patch size')
parser.add_argument('--num_epochs', default=1000, type=int, help='num_epochs')
parser.add_argument('--batch_size', default=1, type=int, help='batch_size per gpu')
parser.add_argument('--val_epochs', default=1, type=int, help='val_epochs')
parser.add_argument('--local_rank', default=0, type=int, help='local rank for DDP')
parser.add_argument('--hafl_warmup_epochs', default=50, type=int, help='HAFL loss warmup epochs (linear ramp-up)')
args = parser.parse_args()

######### DDP Init ###########
dist.init_process_group(backend='nccl')
local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
torch.cuda.set_device(local_rank)
rank = dist.get_rank()
world_size = dist.get_world_size()
is_main = (rank == 0)

if is_main:
    print(f"[DDP] world_size={world_size}, rank={rank}, local_rank={local_rank}")
    print(f"[DDP] 每张卡 batch_size={args.batch_size}, 总 batch_size={args.batch_size * world_size}")

######### SwanLab Init (rank 0 only) ###########
if is_main:
    swanlab.login(api_key="o4MGQAOSX8rGztH69Jj5P")
    swanlab.init(
        project="NeRD-Rain",
        experiment_name="DDP_2GPU_exp3_lr7e5",
        config={
            **vars(args),
            "start_lr": 7e-5,
            "end_lr": 1e-6,
            "warmup_epochs": 0,
            "hafl_warmup_epochs": args.hafl_warmup_epochs,
            "optimizer": "Adam",
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "ddp_world_size": world_size,
            "resume_from": "DDP_2GPU_exp3_lr1e4/model_best.pth",
        },
    )

mode = args.mode
session = args.session
patch_size = args.patch_size

model_dir = os.path.join(args.model_save_dir, mode, 'models', session)
if is_main:
    utils.mkdir(model_dir)

train_dir = args.train_dir
val_dir = args.val_dir

num_epochs = args.num_epochs
batch_size = args.batch_size
val_epochs = args.val_epochs

# 降 LR 续训: 7e-5 → 1e-6
start_lr = 7e-5
end_lr = 1e-6

######### Model ###########
model_restoration = myNet()

if is_main:
    get_parameter_number(model_restoration)

model_restoration = model_restoration.cuda(local_rank)
model_restoration = DDP(model_restoration, device_ids=[local_rank], find_unused_parameters=True)

######### Resume（从原 session 的 best checkpoint 加载） ###########
RESUME = True
RESUME_CKPT_PATH = os.path.join(args.model_save_dir, mode, 'models', 'DDP_2GPU_exp3_lr1e4', 'model_best.pth')

if RESUME:
    # 仅加载模型权重（不加载 optimizer state，使用全新低 LR）
    utils.load_checkpoint(model_restoration.module, RESUME_CKPT_PATH)
    start_epoch = utils.load_start_epoch(RESUME_CKPT_PATH) + 1

    # 动态计算 num_epochs：从 checkpoint epoch 起再跑 500 epoch
    num_epochs = start_epoch + 500 - 1

    if is_main:
        print('------------------------------------------------------------------------------')
        print(f"==> Resume from: {RESUME_CKPT_PATH}")
        print(f"==> Resuming at epoch: {start_epoch}, training until epoch: {num_epochs}")
        print(f"==> New LR: {start_lr} → {end_lr} (CosineAnnealing, no warmup)")
        print('------------------------------------------------------------------------------')

# 创建 optimizer（使用新的低学习率，不恢复旧的 optimizer state）
optimizer = optim.Adam(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler（无 warmup，纯 CosineAnnealing） ###########
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - start_epoch + 1, eta_min=end_lr)

Pretrain = False
model_pre_dir = ''

######### Pretrain ###########
if Pretrain:
    utils.load_checkpoint(model_restoration.module, model_pre_dir)

    if is_main:
        print('------------------------------------------------------------  ------------------')
        print("==> Retrain Training with: " + model_pre_dir)
        print('------------------------------------------------------------------------------')

######### Loss ###########
criterion_char = CharbonnierLoss()
criterion_edge = EdgeLoss()
criterion_fft = fftLoss()
criterion_L1 = nn.L1Loss(size_average=True)
criterion_hafl = HierarchicalAdaptiveFreqLoss()  # 创新点3: HAFL损失

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size': patch_size})
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler,
                          num_workers=4, drop_last=False, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size': patch_size})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                        pin_memory=True)

if is_main:
    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, num_epochs + 1))
    print('===> Loading datasets')

best_psnr = 0
best_epoch = 0
if is_main:
    writer = SummaryWriter(model_dir)
iter = 0

for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_fft_loss = 0
    epoch_char_loss = 0
    epoch_edge_loss = 0
    epoch_l1_loss = 0
    epoch_hafl_loss = 0
    train_id = 1

    # DDP: 每个epoch设置sampler的epoch以保证不同的shuffle
    train_sampler.set_epoch(epoch)

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader, disable=not is_main), 0):

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
        # 创新点3: HAFL损失（对最高尺度输出计算）
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
        iter += 1
        if is_main:
            writer.add_scalar('loss/fft_loss', loss_fft, iter)
            writer.add_scalar('loss/char_loss', loss_char, iter)
            writer.add_scalar('loss/edge_loss', loss_edge, iter)
            writer.add_scalar('loss/l1_loss', loss_l1, iter)
            writer.add_scalar('loss/hafl_loss', loss_hafl, iter)
            writer.add_scalar('loss/iter_loss', loss, iter)

    if is_main:
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

    #### Evaluation (rank 0 only) ####
    if epoch % val_epochs == 0:
        model_restoration.eval()
        if is_main:
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
                            'state_dict': model_restoration.module.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_best.pth"))

            swanlab.log({"val/psnr": psnr_val_rgb, "val/best_psnr": best_psnr, "epoch": epoch}, step=epoch)
            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))


    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
    if is_main:
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                  epoch_loss, current_lr))
        print("------------------------------------------------------------------")
        swanlab.log({"lr": current_lr, "epoch_time": time.time() - epoch_start_time, "epoch": epoch}, step=epoch)

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest.pth"))

if is_main:
    writer.close()
    swanlab.finish()

dist.destroy_process_group()
