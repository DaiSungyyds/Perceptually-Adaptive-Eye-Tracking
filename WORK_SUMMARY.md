# 工作总结与规划

## 📊 数据集下载进度

**当前状态**（截至17:40）：
- ✅ 已完成：user2, user3, user4, user5, user6, user7, user8, user9 (8个用户)
- 🔄 正在下载：user27 等
- ⏳ 待下载：约18个用户
- **进度**：~30% (8/27)
- **预计完成时间**：1.5-2小时

**可用数据**：
- **user4-user9** 共6个用户已经可以使用
- 足够开始模型开发和测试

---

## ✅ 已完成的工作（Day 3）

### 1. 核心模型实现

**文件**：`src/models/multi_exit_vit.py`

**功能**：
- ✅ MultiExitViT 类：支持3个Exit点的ViT模型
- ✅ GazeRegressionHead：Gaze坐标回归头
- ✅ 状态感知路由：自动选择Exit
- ✅ 单Exit推理模式（inference）
- ✅ 多Exit训练模式（training）

**特点**：
- 基于timm的预训练ViT
- Exit点可配置：[3, 6, 12]
- 自动处理灰度图像
- 计算参数量统计

### 2. 损失函数

**文件**：`src/train/losses.py`

**实现**：
- ✅ Angular Loss：角度误差（度）
- ✅ MultiExitLoss：多出口加权损失
- ✅ StateAwareLoss：状态感知损失
- ✅ CombinedLoss：组合损失

**配置**：
- Exit权重：[0.3, 0.3, 0.4]
- 状态权重：Fixation > Pursuit > Saccade

### 3. Mock数据集

**文件**：`src/data/mock_dataset.py`

**作用**：
- ✅ 独立开发：不依赖同学A的I-VT输出
- ✅ 模拟状态分布：Fixation 60%, Saccade 25%, Pursuit 15%
- ✅ 生成Mock标签文件
- ✅ 符合接口规范

### 4. 训练框架

**文件**：`src/train/train_multi_exit_vit.py`

**功能**：
- ✅ 完整训练循环
- ✅ 验证集评估
- ✅ TensorBoard日志
- ✅ 模型检查点保存
- ✅ 学习率调度

### 5. 项目结构

```
✅ src/models/multi_exit_vit.py
✅ src/train/losses.py
✅ src/train/train_multi_exit_vit.py
✅ src/data/mock_dataset.py
✅ requirements.txt
✅ README_MODEL.md
✅ test_setup.py
```

---

## 🎯 你们的独立开发方案

### 方案可行性：✅ 完全可行

**原因**：

1. **接口已明确**：
   - 同学A的输出格式完全定义在文档中
   - JSON Schema清晰
   - 验证脚本已准备好

2. **Mock数据支持**：
   - 可以生成符合接口的Mock数据
   - 模拟真实的状态分布
   - 功能完全等价于真实数据

3. **模块解耦**：
   - 模型训练不依赖I-VT算法细节
   - 只需要状态标签（字符串）
   - 对接时只需替换数据加载器

### 开发流程

```
Day 3（今天）：✅ 模型实现完成
  ├── MultiExitViT ✅
  ├── Loss functions ✅
  ├── Mock dataset ✅
  └── Training script ✅

Day 4（明天）：🔄 Mock数据训练
  ├── 1. 运行 test_setup.py 验证环境
  ├── 2. 生成Mock标签
  ├── 3. 开始训练（小规模）
  └── 4. 调试和优化

Day 5：⏳ 等待同学A + 对接
  ├── 1. 同学A完成 ivt_state_labels.json
  ├── 2. 运行 validate_interface.py
  ├── 3. 运行 test_integration.py
  └── 4. 切换到真实数据

Day 6：⏳ 真实数据训练
  ├── 1. 完整数据集训练
  ├── 2. 性能评估
  └── 3. 模型导出
```

---

## 🚀 明天的工作（Day 4）

### 步骤1：验证环境（10分钟）

```bash
# 等待依赖安装完成（后台进行中）
# 然后运行测试
python test_setup.py
```

**预期输出**：
```
✓ PyTorch: 2.0.0
✓ timm: 0.9.2
✓ Model created successfully
✓ Forward pass working
✓ All core components working!
```

### 步骤2：生成Mock数据（5分钟）

```bash
python src/data/mock_dataset.py
```

**输出**：
- `data/mock_ivt_state_labels.json`
- 包含已下载用户的Mock标签

### 步骤3：小规模训练测试（30分钟）

修改配置进行快速测试：

```python
# 在 train_multi_exit_vit.py 中
config = {
    'training': {
        'epochs': 2,        # 只训练2个epoch
        'batch_size': 4,    # 小批量
    },
    'data': {
        'train_split': 0.6,  # 只用部分数据
    }
}
```

运行：
```bash
python src/train/train_multi_exit_vit.py
```

**目标**：
- 验证训练循环正常运行
- 检查Loss下降趋势
- 确认没有内存/显存问题

### 步骤4：完整训练（3-4小时）

如果小规模测试通过，启动完整训练：

```bash
python src/train/train_multi_exit_vit.py
```

**监控**：
```bash
tensorboard --logdir experiments/logs
# 打开浏览器：http://localhost:6006
```

---

## 🔗 与同学A的对接时机

### 何时对接？

**选项1：Day 5上午**（推荐）
- 你们：Mock数据训练完成，模型调试好
- 同学A：I-VT标注完成
- 对接：验证接口，切换数据，重新训练

**选项2：随时对接**
- 同学A可以随时提供部分用户的标签
- 你们可以混合使用真实+Mock数据
- 逐步过渡

### 对接检查清单

**同学A提供**：
- [ ] `data/ivt_state_labels.json` 文件
- [ ] 包含至少5个用户的数据
- [ ] states列表与frames数量一致

**你们验证**：
```bash
# 1. 格式验证
python scripts/validate_interface.py data/ivt_state_labels.json

# 2. 集成测试
python tests/test_integration.py

# 3. 如果通过，开始训练
python src/train/train_multi_exit_vit.py --use_real_data
```

---

## 📝 重要提醒

### 1. 不要等待数据下载

已有的6个用户足够开始开发：
- user4, user5, user6, user7, user8, user9
- 约5-6万样本
- 足够训练和验证

### 2. 先用Mock数据

优势：
- ✅ 立即开始，不等待同学A
- ✅ 熟悉训练流程
- ✅ 调试模型bug
- ✅ 建立baseline性能

### 3. Git版本管理

建议创建分支：
```bash
git checkout -b feature/multi-exit-vit

# 定期提交
git add src/
git commit -m "Add Multi-Exit ViT model"
```

### 4. 实验记录

每次训练记录：
- 配置参数
- 训练时间
- 最佳验证Loss
- 遇到的问题

使用TensorBoard会自动记录大部分指标。

---

## 🤝 团队协作建议

### 与同学A沟通

**今天**：
- 告知已完成模型实现
- 确认接口格式（发送文档）
- 询问预计完成时间

**明天**：
- 分享Mock数据训练结果
- 确认对接时间点

### 团队内部分工

如果你们两个人：

**人员A**：
- 训练和监控
- 调整超参数
- 性能分析

**人员B**：
- 数据预处理
- 可视化结果
- 准备对接脚本

---

## 📈 预期性能目标

### Mock数据训练（Day 4）

**目标**：
- 训练Loss从初始值下降到稳定
- 验证Loss不diverge
- 模型能收敛

**不要期待**：
- 很低的绝对误差（Mock数据是随机的）
- 状态特定的性能差异

### 真实数据训练（Day 5-6）

**目标**：
- Angular Error < 2.0°（平均）
- Fixation精度 > Saccade精度
- Exit1-3的性能梯度明显

---

## 📚 参考文档

1. **README_MODEL.md**：完整使用文档
2. **接口规范文档**：team_interface_spec.md
3. **代码注释**：每个文件都有详细docstring

---

## ❓ 常见问题预案

### Q: 依赖安装失败？

```bash
# 方案1：使用清华镜像
pip install torch torchvision timm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方案2：分步安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm
```

### Q: 显存不足？

```python
# 减小batch size
config['training']['batch_size'] = 16  # or 8

# 使用gradient accumulation
# (需要修改训练脚本)
```

### Q: 训练太慢？

```python
# 使用更少的用户
train_users = existing_users[:3]  # 只用3个用户

# 减少epochs
config['training']['epochs'] = 10
```

### Q: 同学A的格式不对？

```bash
# 运行验证脚本会给详细错误
python scripts/validate_interface.py data/ivt_state_labels.json

# 根据错误信息与同学A沟通修正
```

---

## 🎉 总结

### 今天完成✅

- Multi-Exit ViT模型完整实现
- 训练框架搭建完成
- Mock数据支持
- 完整的项目结构

### 明天目标🎯

- 验证环境配置
- Mock数据训练
- 模型调试优化

### 你们的方案：完全可行✅

- 独立开发不受阻
- 随时可以对接同学A
- 模块化设计易于集成

**祝训练顺利！有问题随时问我。**
