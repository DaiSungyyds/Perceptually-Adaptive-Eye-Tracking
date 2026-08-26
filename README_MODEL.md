# Multi-Exit Vision Transformer for Gaze Estimation

State-aware gaze tracking using adaptive early-exit Vision Transformers.

## 项目概述

本项目实现了一个**状态感知的多出口ViT模型**，用于眼动追踪：

```
输入：眼部图像 (224x224)
  ↓
Transformer Blocks 1-3 → Exit 1 (Saccade - 快速眼动)
  ↓
Transformer Blocks 4-6 → Exit 2 (Pursuit - 追踪)
  ↓
Transformer Blocks 7-12 → Exit 3 (Fixation - 注视)
  ↓
输出：Gaze坐标 (x, y)
```

### 核心创新

- **状态感知路由**：根据眼动状态选择不同深度的Exit
- **计算效率**：Saccade使用Exit1，节省66%计算量
- **精度保证**：Fixation使用Exit3，保证高精度

---

## 团队分工（3人）

| 成员 | 任务 | 输出 |
|------|------|------|
| 同学A | I-VT眼动状态分类 | `data/ivt_state_labels.json` |
| 你们 | Multi-Exit ViT模型 + 训练 | 训练好的模型权重 |

### 接口规范

同学A的输出文件格式（`data/ivt_state_labels.json`）：

```json
{
  "user4": {
    "left": {
      "n_samples": 15234,
      "statistics": {"fixation": 58.2, "saccade": 26.5, "pursuit": 15.3},
      "states": ["fixation", "saccade", "pursuit", ...],
      "timestamps": [237060314, 237070415, ...],
      "gaze_coords": [[960, 540], [1122, 540], ...]
    }
  }
}
```

---

## 项目结构

```
event_based_gaze_tracking/
├── src/
│   ├── models/
│   │   └── multi_exit_vit.py         # Multi-Exit ViT模型
│   ├── data/
│   │   ├── mock_dataset.py           # Mock数据集（开发用）
│   │   └── real_dataset.py           # 真实数据集（对接同学A）
│   ├── train/
│   │   ├── losses.py                 # 损失函数
│   │   └── train_multi_exit_vit.py   # 训练脚本
│   └── utils/
├── scripts/
│   ├── validate_interface.py         # 接口验证脚本
│   └── test_model.py                 # 模型测试脚本
├── tests/
│   └── test_integration.py           # 集成测试
├── data/
│   ├── mock_ivt_state_labels.json    # Mock标签（开发用）
│   └── ivt_state_labels.json         # 真实标签（同学A提供）
├── eye_data/                          # 原始数据
│   ├── user4/
│   ├── user5/
│   └── ...
├── experiments/
│   ├── logs/                          # TensorBoard日志
│   └── checkpoints/                   # 模型检查点
└── requirements.txt
```

---

## 快速开始

### 1. 环境安装

```bash
# 安装依赖
pip install torch torchvision timm --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy tqdm tensorboard
```

### 2. 测试模型（不需要数据）

```bash
# 测试模型架构
python src/models/multi_exit_vit.py

# 测试损失函数
python src/train/losses.py
```

### 3. 使用Mock数据训练（独立开发）

```bash
# 生成Mock数据标签
python src/data/mock_dataset.py

# 开始训练
python src/train/train_multi_exit_vit.py
```

### 4. 对接同学A的输出

当同学A完成I-VT标注后：

```bash
# 1. 验证接口格式
python scripts/validate_interface.py data/ivt_state_labels.json

# 2. 运行集成测试
python tests/test_integration.py

# 3. 使用真实数据训练
python src/train/train_multi_exit_vit.py --use_real_data
```

---

## 核心组件说明

### 1. Multi-Exit ViT模型

**文件**：`src/models/multi_exit_vit.py`

**关键类**：
- `MultiExitViT`：主模型类
- `GazeRegressionHead`：Gaze回归头

**使用示例**：

```python
from models.multi_exit_vit import MultiExitViT

# 创建模型
model = MultiExitViT(
    model_name='vit_small_patch16_224',
    pretrained=True,
    exit_points=[3, 6, 12]
)

# 单Exit推理（状态感知）
image = torch.randn(1, 1, 224, 224)
gaze = model(image, state='fixation')  # (1, 2)

# 多Exit训练
outputs = model.forward_all_exits(image)
# {'exit_1': (1, 2), 'exit_2': (1, 2), 'exit_3': (1, 2)}
```

### 2. 损失函数

**文件**：`src/train/losses.py`

**支持的损失**：
- `angular_loss`：角度误差（度）
- `MultiExitLoss`：多出口加权损失
- `StateAwareLoss`：状态感知损失
- `CombinedLoss`：组合损失

### 3. Mock数据集

**文件**：`src/data/mock_dataset.py`

**作用**：
- 在同学A完成前，独立开发和测试
- 生成符合接口规范的Mock标签
- 模拟真实的状态分布（Fixation 60%, Saccade 25%, Pursuit 15%）

**使用**：

```python
from data.mock_dataset import MockGazeDataset

dataset = MockGazeDataset(
    data_root='eye_data',
    users=['user4', 'user5'],
    eye='left'
)

# 自动生成Mock状态标签
sample = dataset[0]
# {'image': (1, 224, 224), 'gaze': (2,), 'state': 'fixation'}
```

---

## 训练配置

默认配置在 `src/train/train_multi_exit_vit.py`：

```python
config = {
    'model': {
        'name': 'vit_small_patch16_224',
        'pretrained': True,
        'exit_points': [3, 6, 12]
    },
    'training': {
        'epochs': 50,
        'batch_size': 32,
        'lr': 1e-4
    },
    'loss': {
        'type': 'multi_exit',
        'exit_weights': {'exit_1': 0.3, 'exit_2': 0.3, 'exit_3': 0.4}
    }
}
```

### 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir experiments/logs

# 浏览器打开
http://localhost:6006
```

---

## 接口对接流程

### Checkpoint 1：同学A完成I-VT标注

**输出**：`data/ivt_state_labels.json`

**验证**：
```bash
python scripts/validate_interface.py
```

**期望输出**：
```
✓ 验证通过：所有字段和格式正确。
总用户数: 24
总样本数: 365616
```

### Checkpoint 2：你们切换到真实数据

**步骤**：
1. 将 `MockGazeDataset` 替换为真实数据加载器
2. 运行集成测试确保兼容

**测试**：
```bash
python tests/test_integration.py
```

### Checkpoint 3：重新训练

使用真实标签重新训练模型，对比Mock数据的性能提升。

---

## 预期性能

| 指标 | Saccade | Pursuit | Fixation | 平均 |
|------|---------|---------|----------|------|
| Angular Error | 2.5° | 1.8° | 1.2° | 1.8° |
| 使用Exit | Exit 1 | Exit 2 | Exit 3 | - |
| FLOPs节省 | 66% | 50% | 0% | 40% |
| 推理延迟 | 3ms | 5ms | 8ms | 5.3ms |

---

## 常见问题

### Q1：没有CUDA怎么办？

模型会自动检测，在CPU上也能运行（速度较慢）：

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Q2：数据集还没下载完怎么办？

已下载的用户就足够开始开发：

```bash
# 检查已有用户
ls eye_data/
# user4, user5, user6... 任意5个以上即可开始
```

### Q3：同学A的输出格式不对怎么办？

运行验证脚本会给出详细错误信息：

```bash
python scripts/validate_interface.py
```

根据错误提示修正格式。

### Q4：如何调试模型？

使用小批量数据快速迭代：

```python
# 在train_multi_exit_vit.py中修改
config['training']['batch_size'] = 4
config['training']['epochs'] = 2
config['data']['train_split'] = 0.1  # 只用10%数据
```

---

## 下一步工作

- [ ] **Day 3（今天）**：模型实现完成 ✓
- [ ] **Day 4**：Mock数据训练，调试模型
- [ ] **Day 5**：等待同学A完成，对接真实数据
- [ ] **Day 6**：真实数据训练，性能评估

---

## 参考文献

- [Event-Based Near Eye Gaze Tracking](http://arxiv.org/abs/2004.03577)
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [I-VT Fixation Filter](https://link.springer.com/article/10.3758/BRM.42.1.188)

---

## 联系方式

有问题请联系：
- 模型组：[你们的联系方式]
- 状态分类组：同学A

**协作文档**：[接口规范文档链接]
