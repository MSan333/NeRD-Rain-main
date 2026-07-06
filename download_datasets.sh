#!/bin/bash
# ============================================================
# 去雨数据集下载脚本
# 数据来源：NeRD-Rain 官方 README
# 目标目录：~/pyproject/data/
# ============================================================
#
# 【方案一】百度网盘下载 + scp 上传（推荐，服务器无法直接下载 Google Drive）
#
# Step 1: 在本地电脑下载百度网盘文件：
#   Rain200H: https://pan.baidu.com/s/1KK8R2bPKgcOX8gMXSuKtCQ  提取码: z9br
#   DID-Data: https://pan.baidu.com/s/1aPFJExxxTBOzJjngMAOQDA  提取码: 5luo
#   DDN-Data: https://pan.baidu.com/s/1g_m7RfSUJUtknlWugO1nrw  提取码: ldzo
#
# Step 2: 上传到服务器（在本地终端执行）：
#   scp Rain200H.zip guo_shuaile@<服务器IP>:~/pyproject/data/
#   scp DID-Data.zip guo_shuaile@<服务器IP>:~/pyproject/data/
#   scp DDN-Data.zip guo_shuaile@<服务器IP>:~/pyproject/data/
#
# Step 3: 在服务器上解压：
#   cd ~/pyproject/data
#   unzip Rain200H.zip && rm Rain200H.zip
#   unzip DID-Data.zip && rm DID-Data.zip
#   unzip DDN-Data.zip && rm DDN-Data.zip
#
# -------------------------------------------------------
#
# 【方案二】Google Drive 直接下载（需要服务器能访问 Google，或设置代理）
#   export https_proxy=http://代理地址:端口
#   bash download_datasets.sh --run
#
# 后台运行（断开SSH不中断）：
#   screen -dmS download_data bash download_datasets.sh --run
#   # 查看进度：screen -r download_data
#   # 退出查看：Ctrl+A 然后 D
#
# 【数据集大小估计】
#   Rain200H: ~0.5 GB
#   DID-Data: ~2.5 GB
#   DDN-Data: ~2.0 GB
#   合计约 5 GB
# ============================================================

# 如果没有传 --run 参数，只显示使用说明
if [ "$1" != "--run" ]; then
    echo "请先阅读脚本头部注释的使用说明"
    echo ""
    echo "推荐方案：百度网盘下载 + scp 上传"
    echo ""
    echo "百度网盘链接："
    echo "  Rain200H: https://pan.baidu.com/s/1KK8R2bPKgcOX8gMXSuKtCQ  提取码: z9br"
    echo "  DID-Data: https://pan.baidu.com/s/1aPFJExxxTBOzJjngMAOQDA  提取码: 5luo"
    echo "  DDN-Data: https://pan.baidu.com/s/1g_m7RfSUJUtknlWugO1nrw  提取码: ldzo"
    echo ""
    echo "上传命令（本地终端执行）："
    echo "  scp Rain200H.zip guo_shuaile@<IP>:~/pyproject/data/"
    echo "  scp DID-Data.zip guo_shuaile@<IP>:~/pyproject/data/"
    echo "  scp DDN-Data.zip guo_shuaile@<IP>:~/pyproject/data/"
    echo ""
    echo "服务器解压："
    echo "  cd ~/pyproject/data && unzip Rain200H.zip && unzip DID-Data.zip && unzip DDN-Data.zip"
    echo ""
    echo "如需使用 Google Drive 直接下载，请运行: bash download_datasets.sh --run"
    exit 0
fi

GDOWN=/home/guo_shuaile/.conda/envs/nerd/bin/gdown
DATA_DIR=/home/guo_shuaile/pyproject/data
MAX_RETRY=3  # 每个文件最多重试次数

cd $DATA_DIR || { echo "无法进入 $DATA_DIR"; exit 1; }

# 数据集信息: 名称 Google_Drive_ID
declare -A DATASETS
DATASETS[Rain200H]="1KK8R2bPKgcOX8gMXSuKtCQ"
DATASETS[DID-Data]="1aPFJExxxTBOzJjngMAOQDA"
DATASETS[DDN-Data]="1g_m7RfSUJUtknlWugO1nrw"

DATASET_ORDER=(Rain200H DID-Data DDN-Data)

download_with_retry() {
    local name=$1
    local gdrive_id=$2
    local output="${name}.zip"
    local retry=0

    while [ $retry -lt $MAX_RETRY ]; do
        echo "[尝试 $((retry+1))/$MAX_RETRY] 下载 ${name}..."
        # --continue 支持断点续传（gdown >= 4.6）
        $GDOWN "https://drive.google.com/uc?id=${gdrive_id}" -O "$output" --continue
        if [ $? -eq 0 ] && [ -f "$output" ]; then
            echo "${name} 下载成功!"
            return 0
        fi
        retry=$((retry+1))
        if [ $retry -lt $((MAX_RETRY-1)) ]; then
            echo "下载失败，等待5秒后重试..."
        else
            echo "下载失败"
        fi
        sleep 5
    done
    echo "[错误] ${name} 下载失败，已达最大重试次数"
    return 1
}

extract_dataset() {
    local name=$1
    local zipfile="${name}.zip"

    if [ ! -f "$zipfile" ]; then
        echo "[跳过解压] ${zipfile} 不存在"
        return 1
    fi

    echo "解压 ${name}..."
    unzip -q "$zipfile" -d "${name}_tmp"
    if [ $? -ne 0 ]; then
        echo "[错误] ${name} 解压失败，zip文件可能不完整，删除后重新下载"
        rm -f "$zipfile"
        rm -rf "${name}_tmp"
        return 1
    fi

    # 处理可能的嵌套目录
    if [ -d "${name}_tmp/${name}" ]; then
        mv "${name}_tmp/${name}" ./${name}
        rm -rf "${name}_tmp"
    else
        mv "${name}_tmp" ./${name}
    fi
    rm -f "$zipfile"
    echo "${name} 解压完成!"
    return 0
}

echo "=========================================="
echo "去雨数据集下载脚本（支持断点续传）"
echo "目标目录: $DATA_DIR"
echo "=========================================="

FAILED=()

for name in "${DATASET_ORDER[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "处理: ${name}"
    echo "------------------------------------------"

    # 检查是否已经解压完成
    if [ -d "$DATA_DIR/${name}" ]; then
        echo "[跳过] ${name} 目录已存在，无需重复下载"
        continue
    fi

    # 检查zip是否已下载完成（存在且非空）
    if [ -f "${name}.zip" ] && [ -s "${name}.zip" ]; then
        echo "[发现] ${name}.zip 已存在，尝试直接解压..."
        extract_dataset "$name"
        if [ $? -eq 0 ]; then
            continue
        fi
        echo "解压失败，重新下载..."
    fi

    # 下载
    download_with_retry "$name" "${DATASETS[$name]}"
    if [ $? -ne 0 ]; then
        FAILED+=("$name")
        continue
    fi

    # 解压
    extract_dataset "$name"
    if [ $? -ne 0 ]; then
        FAILED+=("$name")
    fi
done

echo ""
echo "=========================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "全部完成!"
else
    echo "以下数据集处理失败: ${FAILED[*]}"
    echo "请重新运行脚本即可从断点继续"
fi
echo "=========================================="
echo "目录结构："
ls -la $DATA_DIR/
du -sh $DATA_DIR/*/
