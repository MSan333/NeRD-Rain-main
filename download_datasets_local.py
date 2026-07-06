#!/usr/bin/env python3
# ============================================================
# 本地 Windows 下载脚本（支持断点续传）
# 目标目录：D:\data\
#
# 使用前先安装依赖：pip install gdown
#
# 使用方法：
#   python download_datasets_local.py
# ============================================================

import os
import sys
import zipfile
import time
import subprocess

# ====== 配置 ======
DATA_DIR = r"D:\data"
MAX_RETRY = 3

DATASETS = {
    "Rain200H": "1KK8R2bPKgcOX8gMXSuKtCQ",
    "DID-Data": "1aPFJExxxTBOzJjngMAOQDA",
    "DDN-Data": "1g_m7RfSUJUtknlWugO1nrw",
}

DATASET_ORDER = ["Rain200H", "DID-Data", "DDN-Data"]


def download_with_retry(name, gdrive_id):
    """带重试的下载"""
    output = f"{name}.zip"
    for attempt in range(1, MAX_RETRY + 1):
        print(f"\n[尝试 {attempt}/{MAX_RETRY}] 下载 {name}...")
        result = subprocess.run(
            [
                sys.executable, "-m", "gdown",
                f"https://drive.google.com/uc?id={gdrive_id}",
                "-O", output,
                "--continue"
            ],
            cwd=DATA_DIR,
        )
        if result.returncode == 0 and os.path.exists(os.path.join(DATA_DIR, output)):
            print(f"{name} 下载成功!")
            return True
        if attempt < MAX_RETRY:
            print("下载失败，等待5秒后重试...")
            time.sleep(5)
    print(f"[错误] {name} 下载失败，已达最大重试次数")
    return False


def extract_dataset(name):
    """解压数据集"""
    zipfile_path = os.path.join(DATA_DIR, f"{name}.zip")
    tmp_dir = os.path.join(DATA_DIR, f"{name}_tmp")
    target_dir = os.path.join(DATA_DIR, name)

    if not os.path.exists(zipfile_path):
        print(f"[跳过解压] {zipfile_path} 不存在")
        return False

    print(f"解压 {name}...")
    try:
        with zipfile.ZipFile(zipfile_path, "r") as zf:
            zf.extractall(tmp_dir)
    except Exception as e:
        print(f"[错误] {name} 解压失败: {e}")
        os.remove(zipfile_path)
        if os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir)
        return False

    # 处理可能的嵌套目录
    nested = os.path.join(tmp_dir, name)
    if os.path.isdir(nested):
        import shutil
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.move(nested, target_dir)
        shutil.rmtree(tmp_dir)
    else:
        os.rename(tmp_dir, target_dir)

    os.remove(zipfile_path)
    print(f"{name} 解压完成!")
    return True


def main():
    # 创建目标目录
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 50)
    print("去雨数据集下载脚本（支持断点续传）")
    print(f"目标目录: {DATA_DIR}")
    print("=" * 50)

    failed = []

    for name in DATASET_ORDER:
        print(f"\n{'─' * 40}")
        print(f"处理: {name}")
        print(f"{'─' * 40}")

        target_dir = os.path.join(DATA_DIR, name)
        zip_path = os.path.join(DATA_DIR, f"{name}.zip")

        # 已解压完成 → 跳过
        if os.path.isdir(target_dir):
            print(f"[跳过] {name} 目录已存在，无需重复下载")
            continue

        # zip 已存在 → 直接解压
        if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
            print(f"[发现] {name}.zip 已存在，尝试直接解压...")
            if extract_dataset(name):
                continue
            print("解压失败，重新下载...")

        # 下载
        if not download_with_retry(name, DATASETS[name]):
            failed.append(name)
            continue

        # 解压
        if not extract_dataset(name):
            failed.append(name)

    print(f"\n{'=' * 50}")
    if not failed:
        print("全部完成!")
    else:
        print(f"以下数据集处理失败: {failed}")
        print("请重新运行脚本即可从断点继续")
    print(f"{'=' * 50}")

    # 显示目录结构
    if os.path.isdir(DATA_DIR):
        for item in sorted(os.listdir(DATA_DIR)):
            full = os.path.join(DATA_DIR, item)
            if os.path.isdir(full):
                size = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, filenames in os.walk(full)
                    for f in filenames
                )
                print(f"  {item}/  ({size / 1024**3:.2f} GB)")
            else:
                print(f"  {item}  ({os.path.getsize(full) / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
