"""
inference_demo.py — 用最优模型处理几张图片，保存原图和结果到 doc/result/
用于答辩展示
"""
import os
import shutil
import argparse
import torch
import torch.nn as nn
from skimage import img_as_ubyte
from PIL import Image
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import utils
from model import MultiscaleNet as mynet
from layers import window_partitionx, window_reversex

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='./ckpt/Deraininig/models/DDP_4GPU_exp3_lr2e5/model_best.pth', type=str)
parser.add_argument('--input_dir', default='../data/Rain200L/test/input/', type=str)
parser.add_argument('--target_dir', default='../data/Rain200L/test/target/', type=str)
parser.add_argument('--output_dir', default='./doc/result/', type=str)
parser.add_argument('--num_images', default=6, type=int, help='处理图片数量')
parser.add_argument('--win_size', default=256, type=int)
args = parser.parse_args()

# 创建输出目录
os.makedirs(os.path.join(args.output_dir, 'input'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'output'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'target'), exist_ok=True)

# 加载模型
model = mynet()
utils.load_checkpoint(model, args.weights)
print(f"==> 使用模型: {args.weights}")
model.cuda()
model = nn.DataParallel(model)
model.eval()

# 选择代表性图片（均匀采样）
all_images = sorted(os.listdir(args.input_dir))
step = max(1, len(all_images) // args.num_images)
selected = all_images[::step][:args.num_images]
print(f"==> 处理 {len(selected)} 张图片: {selected}")

win = args.win_size

with torch.no_grad():
    for img_name in selected:
        # 读取输入图（带雨）
        input_path = os.path.join(args.input_dir, img_name)
        img = Image.open(input_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()

        # 推理
        _, _, Hx, Wx = input_tensor.shape
        input_re, batch_list = window_partitionx(input_tensor, win)
        restored = model(input_re)
        restored = window_reversex(restored[0], win, Hx, Wx, batch_list)
        restored = torch.clamp(restored, 0, 1)
        restored_np = restored.permute(0, 2, 3, 1).cpu().numpy()[0]
        restored_img = img_as_ubyte(restored_np)

        # 保存
        base_name = os.path.splitext(img_name)[0]
        # 保存输入（带雨图）
        shutil.copy(input_path, os.path.join(args.output_dir, 'input', img_name))
        # 保存去雨结果
        utils.save_img(os.path.join(args.output_dir, 'output', f'{base_name}.png'), restored_img)
        # 保存GT（如果有）
        target_path = os.path.join(args.target_dir, img_name)
        if os.path.exists(target_path):
            shutil.copy(target_path, os.path.join(args.output_dir, 'target', img_name))

        print(f"  [done] {img_name}")

print(f"\n==> 结果已保存到 {args.output_dir}")
print(f"    input/  — 原图（带雨）")
print(f"    output/ — 去雨结果")
print(f"    target/ — GT（无雨参考）")
