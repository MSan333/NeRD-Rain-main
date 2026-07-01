"""
一步完成: 加载模型 → 推理 → 保存结果图 → 计算 Y 通道 PSNR/SSIM

用法:
  python test_and_evaluate.py \
      --weights ./checkpoints/model_best.pth \
      --input_dir ../data/Rain200L/test/input/ \
      --gt_dir ../data/Rain200L/test/target/ \
      --output_dir ./results/Rain200L/ \
      --report_dir ./reports/
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader
from datetime import datetime
from skimage import img_as_ubyte
from tqdm import tqdm
from glob import glob

import utils
from data_RGB import get_test_data
from model import MultiscaleNet as mynet
from get_parameter_number import get_parameter_number
from layers import window_partitionx, window_reversex


# ============================================================
# Y 通道 PSNR/SSIM 计算 (精确复现 Matlab)
# ============================================================

def rgb2ycbcr_y(img_rgb):
    """
    将 RGB uint8 图像转换为 YCbCr 并返回 Y 通道 (uint8)。
    精确复现 Matlab 的 rgb2ycbcr 行为。

    Args:
        img_rgb: numpy array, shape (H, W, 3), dtype uint8, RGB 顺序

    Returns:
        y_channel: numpy array, shape (H, W), dtype uint8
    """
    img = img_rgb.astype(np.float64)
    y = 65.481 / 255.0 * img[:, :, 0] + \
        128.553 / 255.0 * img[:, :, 1] + \
        24.966 / 255.0 * img[:, :, 2] + 16.0
    return np.round(y).clip(0, 255).astype(np.uint8)


def compute_psnr_y(img1, img2):
    """
    计算 Y 通道 PSNR, 复现 Matlab evaluate_PSNR_SSIM.m 中的 compute_psnr。

    Args:
        img1, img2: numpy array, shape (H, W, 3), dtype uint8, RGB 顺序
    """
    y1 = rgb2ycbcr_y(img1).astype(np.float64)
    y2 = rgb2ycbcr_y(img2).astype(np.float64)
    diff = y1 - y2
    rmse = np.sqrt(np.mean(diff ** 2))
    if rmse == 0:
        return float('inf')
    return 20.0 * np.log10(255.0 / rmse)


def compute_ssim_y(img1, img2):
    """
    计算 Y 通道 SSIM, 复现 Matlab evaluate_PSNR_SSIM.m 中的 compute_ssim。
    使用 Matlab SSIM_index 的默认参数: 11x11 Gaussian (sigma=1.5), K=[0.01, 0.03], L=255

    Args:
        img1, img2: numpy array, shape (H, W, 3), dtype uint8, RGB 顺序
    """
    y1 = rgb2ycbcr_y(img1).astype(np.float64)
    y2 = rgb2ycbcr_y(img2).astype(np.float64)
    return _ssim_matlab(y1, y2)


def _ssim_matlab(img1, img2):
    """
    复现 Matlab SSIM_index 函数。
    默认参数: window=fspecial('gaussian',11,1.5), K=[0.01,0.03], L=255
    """
    K1, K2, L = 0.01, 0.03, 255.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    window = _fspecial_gaussian(11, 1.5)

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 * img1, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


def _fspecial_gaussian(size, sigma):
    """复现 Matlab fspecial('gaussian', size, sigma)"""
    x = np.arange(0, size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g = np.outer(g, g)
    return g / g.sum()


# ============================================================
# 主流程
# ============================================================

def save_report(report_dir, weights_path, input_dir, gt_dir, avg_psnr, avg_ssim, count, timestamp):
    """保存评估报告到文本文件"""
    os.makedirs(report_dir, exist_ok=True)
    report_name = f"eval_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = os.path.join(report_dir, report_name)

    lines = [
        "=" * 60,
        "  去雨模型评估报告",
        "=" * 60,
        f"  时间戳:       {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  模型权重:     {weights_path}",
        f"  输入目录:     {input_dir}",
        f"  GT 目录:      {gt_dir}",
        f"  评估图片数:   {count}",
        "",
        f"  Y-channel PSNR: {avg_psnr:.4f} dB",
        f"  Y-channel SSIM: {avg_ssim:.4f}",
        "=" * 60,
    ]
    content = "\n".join(lines) + "\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n评估报告已保存到: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='推理 + Y通道 PSNR/SSIM 评估 (一步完成)')
    parser.add_argument('--weights', required=True, type=str, help='模型权重路径 (必填)')
    parser.add_argument('--input_dir', default='../data/Rain200L/test/input/', type=str, help='测试输入图片目录')
    parser.add_argument('--gt_dir', default='../data/Rain200L/test/target/', type=str, help='GT 目录')
    parser.add_argument('--output_dir', default='./results/Rain200L/', type=str, help='去雨结果保存目录')
    parser.add_argument('--report_dir', default='./reports/', type=str, help='评估报告保存目录')
    parser.add_argument('--gpus', default='0', type=str, help='GPU 编号 (CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--win_size', default=256, type=int, help='window size')
    parser.add_argument('--save_images', default=True, type=lambda x: x.lower() in ('true', '1', 'yes'),
                        help='是否保存结果图片 (默认 True)')
    args = parser.parse_args()

    # ---- 环境设置 ----
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    win = args.win_size
    result_dir = args.output_dir

    # ---- 检查权重文件 ----
    if not os.path.exists(args.weights):
        print(f"错误: 没有找到 ckpt 文件: {args.weights}")
        exit(1)

    # ---- 加载模型 ----
    print(f"===> 加载模型权重: {args.weights}")
    model_restoration = mynet()
    get_parameter_number(model_restoration)
    utils.load_checkpoint(model_restoration, args.weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    # ---- 加载测试数据 ----
    test_dataset = get_test_data(args.input_dir, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                             num_workers=4, drop_last=False, pin_memory=True)

    # ---- 创建输出目录 ----
    if args.save_images:
        utils.mkdir(result_dir)

    # ---- 收集 GT 图片路径 (用于评估) ----
    gt_imgs = sorted(glob(os.path.join(args.gt_dir, '*.png')) +
                     glob(os.path.join(args.gt_dir, '*.jpg')))
    if len(gt_imgs) == 0:
        for alt in ['target', 'targets', 'gt', 'groundtruth']:
            alt_dir = os.path.join(os.path.dirname(args.gt_dir.rstrip('/')), alt)
            gt_imgs = sorted(glob(os.path.join(alt_dir, '*.png')) +
                             glob(os.path.join(alt_dir, '*.jpg')))
            if len(gt_imgs) > 0:
                print(f"在 {alt_dir} 找到 GT 图片")
                break
    gt_dict = {}
    for p in gt_imgs:
        name = os.path.splitext(os.path.basename(p))[0]
        gt_dict[name] = p

    # ---- 推理 + 评估 ----
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    print(f"===> 开始推理 (共 {len(test_loader)} 张图片)...")
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader, desc="推理")):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_test[0].cuda()
            filenames = data_test[1]
            _, _, Hx, Wx = input_.shape

            # 窗口推理
            input_re, batch_list = window_partitionx(input_, win)
            restored = model_restoration(input_re)
            restored = window_reversex(restored[0], win, Hx, Wx, batch_list)

            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                fname = filenames[batch]

                # 保存结果图
                if args.save_images:
                    save_path = os.path.join(result_dir, fname + '.png')
                    utils.save_img(save_path, restored_img)

                # 计算 Y 通道 PSNR/SSIM
                gt_path = gt_dict.get(fname)
                if gt_path is None:
                    continue

                img_gt = cv2.imread(gt_path)
                if img_gt is None:
                    print(f"  警告: 无法读取 GT {gt_path}")
                    continue

                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

                # restored_img 已经是 RGB uint8 (从 tensor permute 得到)
                # 确保尺寸一致
                if restored_img.shape != img_gt.shape:
                    h = min(restored_img.shape[0], img_gt.shape[0])
                    w = min(restored_img.shape[1], img_gt.shape[1])
                    restored_img = restored_img[:h, :w, :]
                    img_gt = img_gt[:h, :w, :]

                psnr_val = compute_psnr_y(restored_img, img_gt)
                ssim_val = compute_ssim_y(restored_img, img_gt)
                total_psnr += psnr_val
                total_ssim += ssim_val
                count += 1

    # ---- 输出结果 ----
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
    else:
        avg_psnr = 0.0
        avg_ssim = 0.0
        print("警告: 没有成功评估任何图片 (GT 匹配失败或 GT 目录为空)")

    print(f"\n{'=' * 50}")
    print(f"  Y-channel PSNR: {avg_psnr:.4f} dB")
    print(f"  Y-channel SSIM: {avg_ssim:.4f}")
    print(f"  评估图片数量:   {count}")
    print(f"{'=' * 50}")

    # ---- 保存报告 ----
    timestamp = datetime.now()
    save_report(args.report_dir, args.weights, args.input_dir, args.gt_dir,
                avg_psnr, avg_ssim, count, timestamp)


if __name__ == '__main__':
    main()
