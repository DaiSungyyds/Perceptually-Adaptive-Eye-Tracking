#!/bin/bash
# 快速启动脚本 - Day 4 使用

echo "=================================="
echo "Multi-Exit ViT 快速启动"
echo "=================================="
echo ""

# Step 1: 验证环境
echo "Step 1: 验证环境..."
python test_setup.py
if [ $? -ne 0 ]; then
    echo "环境验证失败，请先安装依赖"
    exit 1
fi
echo ""

# Step 2: 生成Mock数据
echo "Step 2: 生成Mock数据..."
python src/data/mock_dataset.py
echo ""

# Step 3: 开始训练（小规模测试）
echo "Step 3: 开始训练..."
echo "提示: 首次运行会下载预训练模型（~30MB）"
echo "按 Ctrl+C 可以随时停止"
echo ""
python src/train/train_multi_exit_vit.py

echo ""
echo "=================================="
echo "训练完成！"
echo "查看日志: tensorboard --logdir experiments/logs"
echo "=================================="
