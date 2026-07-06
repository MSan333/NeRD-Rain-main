#!/bin/bash
# ============================================================
# 去雨数据集下载脚本
# 数据来源：NeRD-Rain 官方 README
# 目标目录：~/pyproject/data/
# ============================================================
#
# 【方案一】百度网盘 bypy 命令行下载（推荐，服务器直接下载）
#
# 前置步骤（只需一次）：
#   1. 在浏览器中打开以下链接，将文件转存到自己网盘的 /apps/bypy/ 目录下：
#      Rain200H: https://pan.baidu.com/s/1KK8R2bPKgcOX8gMXSuKtCQ  提取码: z9br
#      DID-Data: https://pan.baidu.com/s/1aPFJExxxTBOzJjngMAOQDA  提取码: 5luo
#      DDN-Data: https://pan.baidu.com/s/1g_m7RfSUJUtknlWugO1nrw  提取码: ldzo
#      注意：转存路径必须是 "我的应用数据/bypy/" 目录下
#   2. 首次运行 bypy 需要授权：
#      /home/guo_shuaile/.conda/envs/nerd/bin/bypy info
#      按提示打开链接授权，粘贴授权码即可
#   3. 授权完成后运行：
#      bash download_datasets.sh --bypy
#
# -------------------------------------------------------
#
# 【方案二】百度网盘本地下载 + scp 上传
#
# Step 1: 在本地电脑下载百度网盘文件（同上链接）
# Step 2: scp 上传到服务器：
#   scp Rain200H.zip guo_shuaile@<服务器IP>:~/pyproject/data/
#   scp DID-Data.zip guo_shuaile@<服务器IP>:~/pyproject/data/
#   scp DDN-Data.zip guo_shuaile@<服务器IP>:~/pyproject/data/
# Step 3: 服务器解压：
#   cd ~/pyproject/data && unzip Rain200H.zip && unzip DID-Data.zip && unzip DDN-Data.zip
#
# -------------------------------------------------------
#
# 【方案三】Google Drive 直接下载（需代理）
#   export https_proxy=http://代理地址:端口
#   bash download_datasets.sh --gdrive
#
# 【数据集大小估计】
#   Rain200H: ~0.5 GB
#   DID-Data: ~2.5 GB
#   DDN-Data: ~2.0 GB
#   合计约 5 GB
#
# 【后台运行】
#   screen -dmS download_data bash download_datasets.sh --bypy
#   screen -r download_data   # 查看进度
#   Ctrl+A D                  # 退出查看
# ============================================================

BYPY=/home/guo_shuaile/.conda/envs/nerd/bin/bypy
GDOWN=/home/guo_shuaile/.conda/envs/nerd/bin/gdown
DATA_DIR=/home/guo_shuaile/pyproject/data
MAX_RETRY=3

mkdir -p $DATA_DIR
cd $DATA_DIR || { echo "无法进入 $DATA_DIR"; exit 1; }

# 数据集列表
DATASET_ORDER=(Rain200H DID-Data DDN-Data)

# Google Drive IDs
declare -A GDRIVE_IDS
GDRIVE_IDS[Rain200H]="1KK8R2bPKgcOX8gMXSuKtCQ"
GDRIVE_IDS[DID-Data]="1aPFJExxxTBOzJjngMAOQDA"
GDRIVE_IDS[DDN-Data]="1g_m7RfSUJUtknlWugO1nrw"

# ============ 公共函数 ============
extract_dataset() {
    local name=$1
    local zipfile="${name}.zip"

    if [ ! -f "$zipfile" ]; then
        echo "[跳过] ${zipfile} 不存在"
        return 1
    fi

    echo "解压 ${name}..."
    unzip -q "$zipfile" -d "${name}_tmp"
    if [ $? -ne 0 ]; then
        echo "[错误] 解压失败，zip可能不完整"
        rm -f "$zipfile"
        rm -rf "${name}_tmp"
        return 1
    fi

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

print_summary() {
    local failed=("$@")
    echo ""
    echo "=========================================="
    if [ ${#failed[@]} -eq 0 ] || [ -z "${failed[0]}" ]; then
        echo "全部完成!"
    else
        echo "失败: ${failed[*]}"
        echo "重新运行脚本即可从断点继续"
    fi
    echo "=========================================="
    echo "目录结构："
    ls -la $DATA_DIR/
    du -sh $DATA_DIR/*/  2>/dev/null
}

# ============ 方案一：bypy 下载 ============
download_bypy() {
    echo "=========================================="
    echo "使用 bypy 从百度网盘下载"
    echo "目标目录: $DATA_DIR"
    echo "=========================================="

    # 检查授权
    $BYPY info 2>&1 | grep -q "Quota"
    if [ $? -ne 0 ]; then
        echo "[错误] bypy 未授权，请先运行: $BYPY info"
        echo "按提示完成授权后重新运行本脚本"
        exit 1
    fi
    echo "bypy 授权正常"

    local FAILED=()
    for name in "${DATASET_ORDER[@]}"; do
        echo ""
        echo "------------------------------------------"
        echo "处理: ${name}"
        echo "------------------------------------------"

        if [ -d "$DATA_DIR/${name}" ]; then
            echo "[跳过] ${name} 目录已存在"
            continue
        fi

        if [ -f "${name}.zip" ] && [ -s "${name}.zip" ]; then
            echo "[发现] ${name}.zip 已存在，尝试解压..."
        else
            echo "从百度网盘下载 ${name}.zip ..."
            $BYPY downfile "${name}.zip" "${name}.zip"
            if [ $? -ne 0 ]; then
                echo "[错误] ${name} 下载失败，请确认文件已转存到 /apps/bypy/ 目录"
                FAILED+=("$name")
                continue
            fi
            echo "${name} 下载成功!"
        fi

        extract_dataset "$name"
        if [ $? -ne 0 ]; then
            FAILED+=("$name")
        fi
    done

    print_summary "${FAILED[@]}"
}

# ============ 方案三：Google Drive 下载 ============
download_gdrive() {
    echo "=========================================="
    echo "使用 gdown 从 Google Drive 下载"
    echo "目标目录: $DATA_DIR"
    echo "=========================================="

    local FAILED=()
    for name in "${DATASET_ORDER[@]}"; do
        echo ""
        echo "------------------------------------------"
        echo "处理: ${name}"
        echo "------------------------------------------"

        if [ -d "$DATA_DIR/${name}" ]; then
            echo "[跳过] ${name} 目录已存在"
            continue
        fi

        if [ -f "${name}.zip" ] && [ -s "${name}.zip" ]; then
            echo "[发现] ${name}.zip 已存在，尝试解压..."
        else
            local retry=0
            local success=0
            while [ $retry -lt $MAX_RETRY ]; do
                echo "[尝试 $((retry+1))/$MAX_RETRY] 下载 ${name}..."
                $GDOWN "https://drive.google.com/uc?id=${GDRIVE_IDS[$name]}" -O "${name}.zip" --continue
                if [ $? -eq 0 ] && [ -f "${name}.zip" ]; then
                    success=1
                    echo "${name} 下载成功!"
                    break
                fi
                retry=$((retry+1))
                echo "下载失败，等待5秒后重试..."
                sleep 5
            done
            if [ $success -eq 0 ]; then
                echo "[错误] ${name} 下载失败"
                FAILED+=("$name")
                continue
            fi
        fi

        extract_dataset "$name"
        if [ $? -ne 0 ]; then
            FAILED+=("$name")
        fi
    done

    print_summary "${FAILED[@]}"
}

# ============ 主逻辑 ============
case "$1" in
    --bypy)
        download_bypy
        ;;
    --gdrive|--run)
        download_gdrive
        ;;
    *)
        echo "去雨数据集下载工具"
        echo ""
        echo "用法:"
        echo "  bash download_datasets.sh --bypy    # 百度网盘命令行下载（推荐）"
        echo "  bash download_datasets.sh --gdrive   # Google Drive 下载（需代理）"
        echo ""
        echo "百度网盘链接（先转存到自己网盘的 /apps/bypy/ 目录）："
        echo "  Rain200H: https://pan.baidu.com/s/1KK8R2bPKgcOX8gMXSuKtCQ  提取码: z9br"
        echo "  DID-Data: https://pan.baidu.com/s/1aPFJExxxTBOzJjngMAOQDA  提取码: 5luo"
        echo "  DDN-Data: https://pan.baidu.com/s/1g_m7RfSUJUtknlWugO1nrw  提取码: ldzo"
        echo ""
        echo "首次使用 bypy 需授权: $BYPY info"
        echo ""
        echo "后台运行: screen -dmS download_data bash download_datasets.sh --bypy"
        ;;
esac
