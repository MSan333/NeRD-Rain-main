#!/bin/bash
# 监控服务器是否恢复（检测 SSH 端口，不用 ping）
# 用法: bash check_server.sh &

HOST="202.117.10.94"
INTERVAL=60  # 每60秒检测一次

echo "开始监控 $HOST SSH端口，每 ${INTERVAL}s 检测一次..."
echo "按 Ctrl+C 停止"

while true; do
    if nc -zv "$HOST" 22 -w 5 2>&1 | grep -q succeeded; then
        echo "✅ $(date): $HOST 已恢复！"

        # macOS 系统通知
        osascript -e "display notification \"服务器 $HOST 已恢复\" with title \"服务器恢复\" sound name \"Glass\""

        # 额外响铃提醒（循环3次）
        for i in 1 2 3; do
            afplay /System/Library/Sounds/Glass.aiff 2>/dev/null
            sleep 1
        done

        echo "通知已发送，脚本结束。"
        exit 0
    else
        echo "❌ $(date): $HOST SSH端口不通，${INTERVAL}s 后重试..."
    fi

    sleep $INTERVAL
done
