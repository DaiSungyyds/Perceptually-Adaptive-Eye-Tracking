# 状态分布配置说明

## 当前配置
```python
state_distribution = {
    'fixation': 0.30,  # 注视 - 眼睛稳定
    'pursuit': 0.60,   # 追踪 - 跟随移动物体  
    'saccade': 0.10    # 快速眼动 - 快速跳转
}
```

## 如何调整

根据你们的实际应用场景调整比例：

### VR游戏场景（当前设置）
- Fixation: 30%
- Pursuit: 60% ⭐ 大量移动追踪
- Saccade: 10%

### VR阅读/文档场景
- Fixation: 70% ⭐ 大量注视文字
- Pursuit: 20%
- Saccade: 10%

### VR搜索/浏览场景
- Fixation: 40%
- Pursuit: 30%
- Saccade: 30% ⭐ 大量快速扫视

## 修改位置

文件: `src/data/mock_dataset.py`

第45-49行:
```python
self.state_distribution = {
    'fixation': 0.30,  # 修改这里
    'pursuit': 0.60,   # 修改这里
    'saccade': 0.10    # 修改这里
}
```

**注意**: 三个值相加必须等于1.0

## 重要提示

当前60% Pursuit的设置是基于：
1. VR/AR的典型使用模式
2. 计算效率和精度的平衡
3. 大部分时间用户在追踪移动内容

如果你们有真实的眼动数据统计，可以根据实际比例调整。
