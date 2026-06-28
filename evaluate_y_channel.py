"""
复现 Matlab evaluate_PSNR_SSIM.m 的 Python 等价脚本。

核心: 在 YCbCr 色彩空间的 Y 通道上计算 PSNR 和 SSIM,
      与论文报告的指标完全一致。

用法:
  python evaluate_y_channel.py \
      --result_dir ./results/Rain200L \
      --gt_dir ./Datasets/Rain200L/test/target

  或直接修改下面的默认路径运行:
  python evaluate_y_channel.py
"""

import os
import argparse
import numpy as np
import cv2
from glob import glob


# ============================================================
# 精确复现 Matlab rgb2ycbcr (uint8 版本)
# ============================================================
# Matlab 的 rgb2ycbcr 对 uint8 输入使用以下公式:
#   Y  =  65.481 * R/255 + 128.553 * G/255 +  24.966 * B/255 +  16
#   Cb = -37.797 * R/255 -  74.203 * G/255 + 112.0   * B/255 + 128
#   Cr = 112.0   * R/255 -  93.786 * G/255 -  18.214 * B/255 + 128
# 输出 uint8, 范围 [16, 235] (Y) / [16, 240] (Cb, Cr)
#
# 简化后只取 Y 通道:
#   Y = 0.256789 * R + 0.504129 * G + 0.097906 * B + 16
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

    # fspecial('gaussian', 11, 1.5)
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


def main():
    parser = argparse.ArgumentParser(description='Y-channel PSNR/SSIM (复现 Matlab 评估)')
    parser.add_argument('--result_dir', type=str,
                        default='./evaluations/Evalution_Rain200L_Rain200H_SPA-Data/results/Rain200L',
                        help='去雨结果图目录')
    parser.add_argument('--gt_dir', type=str,
                        default='../data/Rain200L/test/target',
                        help='Ground truth 目录')
    args = parser.parse_args()

    # 收集图片 (支持 .png 和 .jpg)
    result_imgs = sorted(glob(os.path.join(args.result_dir, '*.png')) +
                         glob(os.path.join(args.result_dir, '*.jpg')))
    gt_imgs = sorted(glob(os.path.join(args.gt_dir, '*.png')) +
                     glob(os.path.join(args.gt_dir, '*.jpg')))

    if len(result_imgs) == 0:
        print(f"错误: {args.result_dir} 下没有找到图片")
        return
    if len(gt_imgs) == 0:
        # 尝试常见的其他 GT 目录名
        for alt in ['target', 'targets', 'gt', 'groundtruth']:
            alt_dir = os.path.join(os.path.dirname(args.gt_dir), alt)
            gt_imgs = sorted(glob(os.path.join(alt_dir, '*.png')) +
                             glob(os.path.join(alt_dir, '*.jpg')))
            if len(gt_imgs) > 0:
                print(f"在 {alt_dir} 找到 GT 图片")
                break
        if len(gt_imgs) == 0:
            print(f"错误: {args.gt_dir} 下没有找到 GT 图片")
            return

    print(f"结果图: {len(result_imgs)} 张  ({args.result_dir})")
    print(f"GT 图:  {len(gt_imgs)} 张  ({args.gt_dir})")

    if len(result_imgs) != len(gt_imgs):
        print(f"警告: 结果图数量 ({len(result_imgs)}) != GT 数量 ({len(gt_imgs)})")
        # 尝试按文件名匹配
        gt_dict = {}
        for p in gt_imgs:
            name = os.path.splitext(os.path.basename(p))[0]
            gt_dict[name] = p

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for i, result_path in enumerate(result_imgs):
        result_name = os.path.splitext(os.path.basename(result_path))[0]

        # 匹配 GT
        if len(result_imgs) == len(gt_imgs):
            gt_path = gt_imgs[i]
        else:
            gt_path = gt_dict.get(result_name)
            if gt_path is None:
                print(f"  跳过 {result_name}: 未找到对应 GT")
                continue

        # 读取图片 (cv2 读取为 BGR, 转为 RGB)
        img_result = cv2.imread(result_path)
        img_gt = cv2.imread(gt_path)

        if img_result is None:
            print(f"  跳过: 无法读取 {result_path}")
            continue
        if img_gt is None:
            print(f"  跳过: 无法读取 {gt_path}")
            continue

        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        # 确保尺寸一致
        if img_result.shape != img_gt.shape:
            h = min(img_result.shape[0], img_gt.shape[0])
            w = min(img_result.shape[1], img_gt.shape[1])
            img_result = img_result[:h, :w, :]
            img_gt = img_gt[:h, :w, :]

        psnr_val = compute_psnr_y(img_result, img_gt)
        ssim_val = compute_ssim_y(img_result, img_gt)
        total_psnr += psnr_val
        total_ssim += ssim_val
        count += 1

    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print(f"\n{'='*50}")
        print(f"  Y-channel PSNR: {avg_psnr:.4f} dB")
        print(f"  Y-channel SSIM: {avg_ssim:.4f}")
        print(f"  评估图片数量:   {count}")
        print(f"{'='*50}")
        print(f"\n这个结果应与 Matlab evaluate_PSNR_SSIM.m 一致,")
        print(f"即论文中报告的指标。")
    else:
        print("没有成功评估任何图片")


if __name__ == '__main__':
    main()
