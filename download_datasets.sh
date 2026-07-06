#!/bin/bash
# ============================================================
# 去雨数据集下载脚本（支持断点续传）
# 数据来源：NeRD-Rain 官方 README 中的 Google Drive 链接
# ============================================================
#
# 【使用说明】
# 1. 需要在能访问 Google Drive 的环境（本地电脑/有代理的机器）运行
# 2. 需要先安装 gdown：pip install gdown
# 3. 运行方式：
#      chmod +x download_datasets.sh
#      bash download_datasets.sh
# 4. 中断后重新运行即可从断点继续（已下载的不会重复下载）
# 5. 下载完成后将数据上传到服务器：
#      scp -r ./data/Rain200H guo_shuaile@<服务器IP>:~/pyproject/data/
#      scp -r ./data/DID-Data guo_shuaile@<服务器IP>:~/pyproject/data/
#      scp -r ./data/DDN-Data guo_shuaile@<服务器IP>:~/pyproject/data/
#    或者打包后上传：
#      tar czf datasets.tar.gz ./data/Rain200H ./data/DID-Data ./data/DDN-Data
#      scp datasets.tar.gz guo_shuaile@<服务器IP>:~/pyproject/
#      # 在服务器上解压：tar xzf datasets.tar.gz
#
# 【数据集大小估计】
#   Rain200H: ~0.5 GB
#   DID-Data: ~2.5 GB
#   DDN-Data: ~2.0 GB
#   合计约 5 GB
# ============================================================

# gdown 路径（本地运行时改为 gdown 即可，需 pip install gdown）
GDOWN=${GDOWN_BIN:-gdown}
DATA_DIR=${DATA_DIR:-./data}
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
        echo "下载失败，${retry < MAX_RETRY:+等待5秒后重试...}"
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
