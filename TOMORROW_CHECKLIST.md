# 明天的操作清单 (Day 4)

## ✅ 待办事项

### 第一步：验证环境 (10分钟)
```bash
# 确保依赖安装完成
python test_setup.py
```
**预期输出**: ✓ PyTorch, ✓ timm, ✓ 所有组件正常

---

### 第二步：验证模型 (10分钟)
```bash
# 运行完整验证脚本
python verify_model.py
```
**预期输出**: ✓✓✓ 所有测试通过！

**7项测试内容**:
- [x] 模型创建与架构
- [x] 单Exit推理模式
- [x] 多Exit训练模式
- [x] 计算效率验证
- [x] 损失函数
- [x] 反向传播
- [x] 状态路由

---

### 第三步：生成Mock数据 (5分钟)
```bash
# 生成模拟的状态标签
python src/data/mock_dataset.py
```
**输出**: `data/mock_ivt_state_labels.json`

**状态分布**:
- Fixation: 30%
- Pursuit: 60% ⭐
- Saccade: 10%

---

### 第四步：小规模测试 (30分钟)

**修改配置** (`src/train/train_multi_exit_vit.py`):
```python
config = {
    'training': {
        'epochs': 2,          # 只训练2轮
        'batch_size': 4,      # 小批量
    },
    'data': {
        'train_split': 0.6,   # 只用60%数据
    }
}
```

**运行**:
```bash
python src/train/train_multi_exit_vit.py
```

**目标**: 验证训练循环能正常运行

---

### 第五步：完整训练 (3-4小时)

**恢复配置**:
```python
config = {
    'training': {
        'epochs': 50,
        'batch_size': 32,
    },
    'data': {
        'train_split': 0.7,
    }
}
```

**运行训练**:
```bash
python src/train/train_multi_exit_vit.py
```

**监控训练**:
```bash
# 新开一个终端
tensorboard --logdir experiments/logs

# 浏览器打开
# http://localhost:6006
```

---

## 📊 检查点

### Checkpoint 1: 环境验证通过
- [ ] PyTorch正常
- [ ] timm正常
- [ ] 模型能创建

### Checkpoint 2: 模型验证通过
- [ ] 7项测试全部通过
- [ ] 无NaN/错误
- [ ] 参数能更新

### Checkpoint 3: Mock数据生成
- [ ] JSON文件生成
- [ ] 格式正确
- [ ] 15个用户数据

### Checkpoint 4: 小规模训练成功
- [ ] 训练循环正常
- [ ] Loss下降
- [ ] 无显存溢出

### Checkpoint 5: 完整训练启动
- [ ] 训练正常运行
- [ ] TensorBoard可视化
- [ ] 定期保存checkpoint

---

## 🚨 常见问题处理

### 问题1: CUDA out of memory
**解决**:
```python
config['training']['batch_size'] = 16  # 或更小
```

### 问题2: 训练很慢
**解决**:
```python
# 减少数据量
train_users = existing_users[:5]  # 只用5个用户

# 减少epochs
config['training']['epochs'] = 20
```

### 问题3: Loss不下降
**检查**:
- Learning rate是否合适 (1e-4)
- 数据是否正确加载
- 查看TensorBoard曲线

---

## 📝 记录模板

### 训练记录
```
日期: 2024-XX-XX
配置:
  - Epochs: 50
  - Batch Size: 32
  - Learning Rate: 1e-4
  - 用户数: 15

结果:
  - 最佳Val Loss: ____
  - 训练时间: ____
  - Exit 1 Loss: ____
  - Exit 2 Loss: ____
  - Exit 3 Loss: ____

问题:
  - (记录遇到的问题)

解决方案:
  - (记录如何解决)
```

---

## 📚 快速参考

### 关键文件
```
src/models/multi_exit_vit.py       # 模型定义
src/train/train_multi_exit_vit.py # 训练脚本
src/train/losses.py                # 损失函数
src/data/mock_dataset.py           # Mock数据
verify_model.py                    # 验证脚本
```

### 关键文档
```
README_MODEL.md              # 使用指南
ARCHITECTURE_EXPLAINED.md    # 架构详解
WORK_SUMMARY.md             # 工作总结
```

### 有用的命令
```bash
# 查看GPU使用
nvidia-smi

# 查看训练进程
ps aux | grep python

# 查看日志
tail -f experiments/logs/*/events.out.*

# 停止训练
Ctrl+C
```

---

## ✨ 成功标志

### 小规模训练成功
- ✅ 2个epoch完成
- ✅ Loss从初始值下降
- ✅ 无错误或崩溃
- ✅ 模型checkpoint保存

### 完整训练成功
- ✅ 50个epoch完成
- ✅ Val Loss收敛
- ✅ TensorBoard曲线平滑
- ✅ Best model保存

---

## 🎯 明天结束时的目标

- [x] 环境完全配置好
- [x] 模型验证通过
- [x] Mock数据训练完成
- [x] 建立性能baseline
- [x] 熟悉训练流程

**如果一切顺利，Day 5可以直接对接同学A的真实数据！**

---

## 📞 需要帮助？

1. 查看文档: `ARCHITECTURE_EXPLAINED.md`
2. 运行测试: `python verify_model.py`
3. 检查日志: TensorBoard

**祝训练顺利！🚀**
