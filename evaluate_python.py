import cv2
import glob
import math
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim


def rgb2ycbcr(img):
    """
    Convert RGB image to YCbCr color space.
    Matches MATLAB's rgb2ycbcr implementation.
    """
    ycbcr = np.empty_like(img, dtype=np.float32)
    ycbcr[:, :, 0] = 16.0 + (65.481 * img[:, :, 0] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 2])
    ycbcr[:, :, 1] = 128.0 + (-37.797 * img[:, :, 0] - 74.203 * img[:, :, 1] + 112.0 * img[:, :, 2])
    ycbcr[:, :, 2] = 128.0 + (112.0 * img[:, :, 0] - 93.786 * img[:, :, 1] - 18.214 * img[:, :, 2])
    return ycbcr

def compute_psnr(img1, img2):
    """
    Compute PSNR on the Y channel.
    """
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1 = rgb2ycbcr(img1 / 255.0)
        img1 = img1[:, :, 0]

    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2 = rgb2ycbcr(img2 / 255.0)
        img2 = img2[:, :, 0]

    img1 = np.float64(img1)
    img2 = np.float64(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def compute_ssim(img1, img2):
    """
    Compute SSIM on the Y channel.
    """
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1 = rgb2ycbcr(img1 / 255.0)
        img1 = img1[:, :, 0]

    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2 = rgb2ycbcr(img2 / 255.0)
        img2 = img2[:, :, 0]

    # Convert to uint8 for skimage ssim
    img1 = np.round(img1).astype(np.uint8)
    img2 = np.round(img2).astype(np.uint8)

    # Use skimage's SSIM which is very close to MATLAB's
    score, _ = ssim(img1, img2, full=True, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return score

def main():
    datasets = ['Rain200L']
    # datasets = ['Rain200L', 'Rain200H', 'SPA-Data']

    psnr_alldatasets = 0
    ssim_alldatasets = 0

    for dataset in datasets:
        file_path = os.path.join('./evaluations/Evalution_Rain200L_Rain200H_SPA-Data/results', dataset)
        gt_path = os.path.join('./evaluations/Evalution_Rain200L_Rain200H_SPA-Data/Datasets', dataset, 'targets')

        # Get all jpg and png files
        path_list = glob.glob(os.path.join(file_path, '*.jpg')) + glob.glob(os.path.join(file_path, '*.png'))
        path_list.sort()

        img_num = len(path_list)

        if img_num == 0:
            print(f"No images found in {file_path}")
            continue

        total_psnr = 0
        total_ssim = 0

        for img_path in path_list:
            image_name = os.path.basename(img_path)
            gt_img_path = os.path.join(gt_path, image_name)

            if not os.path.exists(gt_img_path):
                print(f"Warning: GT not found for {image_name}")
                continue

            # Read images (cv2 reads in BGR, convert to RGB)
            input_img = cv2.imread(img_path)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            gt_img = cv2.imread(gt_img_path)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

            psnr_val = compute_psnr(input_img, gt_img)
            ssim_val = compute_ssim(input_img, gt_img)

            total_psnr += psnr_val
            total_ssim += ssim_val

        qm_psnr = total_psnr / img_num
        qm_ssim = total_ssim / img_num

        print(f'For {dataset} dataset PSNR: {qm_psnr:.4f} SSIM: {qm_ssim:.4f}')

        psnr_alldatasets += qm_psnr
        ssim_alldatasets += qm_ssim

    if len(datasets) > 0:
        print(f'For all datasets PSNR: {psnr_alldatasets/len(datasets):.4f} SSIM: {ssim_alldatasets/len(datasets):.4f}')

if __name__ == '__main__':
    main()

