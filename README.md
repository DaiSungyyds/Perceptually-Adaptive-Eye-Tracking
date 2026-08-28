# Event-Based Gaze Tracking

基于事件相机的眼球追踪项目，使用RVT (Recurrent Vision Transformer)模型。

## 项目概述

本项目实现了基于事件相机数据的眼球注视点预测，通过向量化优化实现了高效训练。

## 主要特性

- **向量化数据处理**：events_to_polarity_image函数采用PyTorch scatter_优化，相比原始Python循环版本提速174.6倍
- **高效训练**：单epoch训练时间约1.2分钟（相比未优化版本的40-50分钟）
- **Pixel Loss**：使用欧氏距离损失函数，直接优化屏幕坐标误差

## 性能指标

- **最佳验证损失**: 10.26 pixels (Epoch 92/100)
- **训练时间**: 约2小时完成100 epochs
- **训练速度**: 13-14 iterations/秒

## 环境配置

```bash
pip install -r requirements.txt
```

## 数据集

使用预处理的pkl缓存数据：
- 训练集: LREye_train_cache_drop15_8000acc (251,328样本)
- 验证集: LREye_val_cache_drop15_8000acc (10,048样本)
- Events数量: 8000 events/样本

## 训练

```bash
# 在服务器上训练
export PYTHONPATH=/path/to/project:/path/to/project/new_dataset:$PYTHONPATH
python3 train_with_optloader.py
```

配置参数：
- Batch Size: 256
- Learning Rate: 1e-4
- Epochs: 100
- GPU: NVIDIA A100 80GB

## 模型文件

训练好的模型权重：
- 最佳模型: `best_model_gpu3_optloader.pth` (10.26 pixels)
- 由于文件较大(9.7MB)，未上传到GitHub
- 可从[模型托管链接]下载

## 关键优化

### 向量化events_to_polarity_image

原始版本使用Python for循环逐个event处理：
```python
for event in events:
    x, y = int(x.item()), int(y.item())
    polarity_img[y, x] = 1 if p.item() == 1 else -1
```

优化版本使用PyTorch scatter_批量处理：
```python
flat_indices = y_valid * width + x_valid
flat_img.scatter_(0, flat_indices, polarity_values)
```

性能提升：
- 原始版本: 63.08ms/样本
- 向量化版本: 0.36ms/样本
- **加速比: 174.6x**

## 项目结构

```
event_based_gaze_tracking/
├── train_with_optloader.py      # 训练脚本（向量化优化版）
├── test_vectorization.py        # 向量化验证脚本
├── eval_pixel_error.py          # 评估脚本
├── src/
│   └── models/
│       └── rvt_gaze.py         # RVT模型定义
├── new_dataset/
│   └── sel_valid_label_0815/   # 数据加载模块
└── requirements.txt
```

## 训练结果

| Metric | Value |
|--------|-------|
| 最终Train Loss | 36.24 pixels |
| 最佳Val Loss | 10.26 pixels |
| 训练时间/epoch | ~1.2分钟 |
| 总训练时间 | ~2小时 |

## 许可证

[添加许可证信息]

## 致谢

- 数据集和baseline来自师兄的工作
- 使用LoadFromCache数据加载器
