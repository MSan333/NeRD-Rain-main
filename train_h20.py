import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from model import MultiscaleNet as myNet
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from get_parameter_number import get_parameter_number
import kornia
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import argparse
import swanlab

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1

parser = argparse.ArgumentParser(description='Image Deraining - H20 Adapted')

parser.add_argument('--train_dir', default='../data/Rain200L/train/', type=str, help='Directory of train images')
parser.add_argument('--val_dir', default='../data/Rain200L/test/', type=str, help='Directory of validation images')
parser.add_argument('--model_save_dir', default='./checkpoints/', type=str, help='Path to save weights')
parser.add_argument('--pretrain_weights', default='', type=str, help='Path to pretrain weights for curriculum stage')
parser.add_argument('--mode', default='Deraininig', type=str)
parser.add_argument('--session', default='Multiscale_H20', type=str, help='session name')
parser.add_argument('--patch_size', default=256, type=int, help='patch size')
parser.add_argument('--num_epochs', default=3000, type=int, help='num epochs')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--val_epochs', default=1, type=int, help='validation frequency')
# H20 specific args
parser.add_argument('--start_lr', default=8e-4, type=float, help='initial learning rate')
parser.add_argument('--end_lr', default=8e-6, type=float, help='minimum learning rate')
parser.add_argument('--warmup_epochs', default=3, type=int, help='warmup epochs')
parser.add_argument('--num_workers', default=16, type=int, help='dataloader workers')
parser.add_argument('--no_amp', action='store_true', help='disable mixed precision training')
parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint in model_save_dir')
parser.add_argument('--alpha_lr_multiplier', default=10, type=float, help='MDPConv alpha params LR multiplier (MFGCP)')
args = parser.parse_args()

######### SwanLab Init ###########
try:
    swanlab.login(api_key="o4MGQAOSX8rGztH69Jj5P")
    swanlab.init(
        project="NeRD-Rain",
        experiment_name=args.session,
        config={
            **vars(args),
            "optimizer": "Adam",
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "mixed_precision": "bfloat16" if not args.no_amp else "none",
        },
    )
    SWANLAB_ENABLED = True
except Exception as e:
    print(f"[WARNING] SwanLab init failed: {e}. Training without SwanLab.")
    SWANLAB_ENABLED = False

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
start_lr = args.start_lr
end_lr = args.end_lr
warmup_epochs = args.warmup_epochs
use_amp = not args.no_amp

######### Model ###########
model_restoration = myNet()
get_parameter_number(model_restoration)
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print(f"\nUsing {torch.cuda.device_count()} GPUs!\n")

# Separate alpha params for higher LR (MFGCP doc Section 5.1)
alpha_params = []
other_params = []
for n, p in model_restoration.named_parameters():
    if 'alpha' in n:
        alpha_params.append(p)
    else:
        other_params.append(p)

param_groups = [{'params': other_params, 'lr': start_lr}]
if alpha_params:
    param_groups.append({'params': alpha_params, 'lr': start_lr * args.alpha_lr_multiplier})
    print(f"[MFGCP] Alpha params: {len(alpha_params)} tensors, LR={start_lr * args.alpha_lr_multiplier:.6f}")

optimizer = optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler ###########
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - warmup_epochs, eta_min=end_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

######### Pretrain / Resume ###########
if args.pretrain_weights and os.path.exists(args.pretrain_weights):
    checkpoint = torch.load(args.pretrain_weights, map_location='cpu')
    # Handle DataParallel 'module.' prefix
    state_dict = checkpoint['state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model_restoration.load_state_dict(state_dict)
    print(f'==> Loaded pretrain weights: {args.pretrain_weights}')
    print(f'    From epoch {checkpoint["epoch"]}')
elif args.resume:
    try:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)
        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print(f'==> Resumed from epoch {start_epoch - 1}, lr: {new_lr}')
    except Exception as e:
        print(f'[WARNING] Resume failed: {e}. Starting from scratch.')
        start_epoch = 1

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_fft = losses.fftLoss()
criterion_L1 = nn.L1Loss(size_average=True)

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size': patch_size})
loader_kwargs = {
    'batch_size': batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'drop_last': False,
    'pin_memory': True,
    'persistent_workers': args.num_workers > 0,
}
if args.num_workers > 0:
    loader_kwargs['prefetch_factor'] = 4
train_loader = DataLoader(dataset=train_dataset, **loader_kwargs)

val_dataset = get_validation_data(val_dir, {'patch_size': patch_size})
val_loader_kwargs = {
    'batch_size': 1,
    'shuffle': False,
    'num_workers': min(4, args.num_workers),
    'drop_last': False,
    'pin_memory': True,
    'persistent_workers': min(4, args.num_workers) > 0,
}
val_loader = DataLoader(dataset=val_dataset, **val_loader_kwargs)

print(f'===> Start Epoch {start_epoch} End Epoch {num_epochs + 1}')
print(f'===> Config: ps={patch_size}, bs={batch_size}, lr={start_lr}->{end_lr}, '
      f'warmup={warmup_epochs}, amp={use_amp}, workers={args.num_workers}')
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0
writer = SummaryWriter(model_dir)
iter_count = 0
scaler = GradScaler(enabled=use_amp)

amp_dtype = torch.bfloat16

for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_fft = 0
    epoch_char = 0
    epoch_edge = 0
    epoch_l1 = 0
    num_iters = 0

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        for param in model_restoration.parameters():
            param.grad = None

        target_ = data[0].cuda(non_blocking=True)
        input_ = data[1].cuda(non_blocking=True)

        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            target = kornia.geometry.transform.build_pyramid(target_, 3)
            restored = model_restoration(input_)

            loss_fft = criterion_fft(restored[0], target[0]) + criterion_fft(restored[1], target[1]) + criterion_fft(restored[2], target[2])
            loss_char = criterion_char(restored[0], target[0]) + criterion_char(restored[1], target[1]) + criterion_char(restored[2], target[2])
            loss_edge = criterion_edge(restored[0], target[0]) + criterion_edge(restored[1], target[1]) + criterion_edge(restored[2], target[2])
            loss_l1 = criterion_L1(restored[3], target[1]) + criterion_L1(restored[5], target[2])
            loss = loss_char + 0.01 * loss_fft + 0.05 * loss_edge + 0.1 * loss_l1

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        epoch_fft += loss_fft.item()
        epoch_char += loss_char.item()
        epoch_edge += loss_edge.item()
        epoch_l1 += loss_l1.item()
        num_iters += 1
        iter_count += 1

        # iter 级 loss 只记录到 tensorboard（swanlab 统一用 epoch 做 x 轴）
        writer.add_scalar('loss/fft_loss', loss_fft, iter_count)
        writer.add_scalar('loss/char_loss', loss_char, iter_count)
        writer.add_scalar('loss/edge_loss', loss_edge, iter_count)
        writer.add_scalar('loss/l1_loss', loss_l1, iter_count)
        writer.add_scalar('loss/iter_loss', loss, iter_count)

    if SWANLAB_ENABLED:
        swanlab.log({
            "loss/epoch_loss": epoch_loss / num_iters,
            "loss/fft_loss": epoch_fft / num_iters,
            "loss/char_loss": epoch_char / num_iters,
            "loss/edge_loss": epoch_edge / num_iters,
            "loss/l1_loss": epoch_l1 / num_iters,
        }, step=epoch)
    writer.add_scalar('loss/epoch_loss', epoch_loss, epoch)

    #### Evaluation ####
    if epoch % val_epochs == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda(non_blocking=True)
            input_ = data_val[1].cuda(non_blocking=True)

            with torch.no_grad():
                restored = model_restoration(input_)

            for res, tar in zip(restored[0], target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        writer.add_scalar('val/psnr', psnr_val_rgb, epoch)
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_dir, "model_best.pth"))

        if SWANLAB_ENABLED:
            swanlab.log({"val/psnr": psnr_val_rgb, "val/best_psnr": best_psnr}, step=epoch)
        print(f"[epoch {epoch} PSNR: {psnr_val_rgb:.4f} --- best_epoch {best_epoch} Best_PSNR {best_psnr:.4f}]")

    scheduler.step()

    current_lr = scheduler.get_lr()[0]
    elapsed = time.time() - epoch_start_time
    print(f"Epoch: {epoch}\tTime: {elapsed:.2f}s\tLoss: {epoch_loss:.4f}\tLR: {current_lr:.6f}")
    if SWANLAB_ENABLED:
        swanlab.log({"lr": current_lr, "epoch_time": elapsed}, step=epoch)

    torch.save({
        'epoch': epoch,
        'state_dict': model_restoration.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(model_dir, "model_latest.pth"))

writer.close()
if SWANLAB_ENABLED:
    swanlab.finish()
print(f"\nTraining complete! Best PSNR: {best_psnr:.4f} at epoch {best_epoch}")
