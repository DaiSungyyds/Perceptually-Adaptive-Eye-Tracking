# Multi-Exit ViT 架构详解

## 📊 一、整体架构图

```
输入: 眼部图像 (Batch, 1, 224, 224)
│
├─> [自动复制为3通道] (B, 3, 224, 224)
│
├─> [Patch Embedding Layer]
│   ├─ 16×16 patches
│   └─> (B, 196, 384) + CLS token → (B, 197, 384)
│
├─> [Position Encoding]
│   └─> 每个token添加位置信息
│
├─> [Transformer Encoder]
│   │
│   ├─> Block 1  ┐
│   ├─> Block 2  │ Stage 1: Early Features
│   ├─> Block 3  ┘
│   │   │
│   │   └─────> [Exit 1: Early] ──────────────┐
│   │           │ GazeHead_1(384 → 256 → 2)   │
│   │           │ 用于: Saccade (快速眼动)     │
│   │           └─> (B, 2) 坐标输出            │
│   │                                          │
│   ├─> Block 4  ┐                            │
│   ├─> Block 5  │ Stage 2: Medium Features   │
│   ├─> Block 6  ┘                            │
│   │   │                                     │
│   │   └─────> [Exit 2: Medium] ────────────┤
│   │           │ GazeHead_2(384 → 256 → 2)   │
│   │           │ 用于: Pursuit (追踪)         │
│   │           └─> (B, 2) 坐标输出            │
│   │                                          │
│   ├─> Block 7  ┐                            │
│   ├─> Block 8  │                            │
│   ├─> Block 9  │ Stage 3: Deep Features     │
│   ├─> Block 10 │                            │
│   ├─> Block 11 │                            │
│   ├─> Block 12 ┘                            │
│       │                                     │
│       └─────> [Exit 3: Deep] ───────────────┘
│               │ GazeHead_3(384 → 256 → 2)
│               │ 用于: Fixation (注视)
│               └─> (B, 2) 坐标输出
│
└─> 训练时: 3个输出同时优化
    推理时: 根据状态选择1个Exit
```

---

## 🧩 二、核心组件详解

### 2.1 Patch Embedding

**作用**: 将图像转换为token序列

```python
输入图像: (B, 1, 224, 224) 灰度图
    ↓ 复制3通道
(B, 3, 224, 224)
    ↓ 切分为16×16 patches
(B, 196, 3×16×16) = (B, 196, 768)
    ↓ 线性投影到384维
(B, 196, 384)
    ↓ 添加CLS token
(B, 197, 384)
```

**为什么用Patch?**
- ViT的核心思想: 图像 = 序列
- 16×16是标准配置
- 224×224 ÷ 16 = 14×14 = 196个patches

### 2.2 Transformer Block

**每个Block包含两个子层:**

```
输入 x (B, 197, 384)
    ↓
┌───────────────────────────┐
│ 1. Multi-Head Attention   │
│    ├─ Query, Key, Value   │
│    ├─ 12个attention heads  │
│    └─ Self-Attention      │
└───────────────────────────┘
    ↓ + Residual
    ↓ LayerNorm
┌───────────────────────────┐
│ 2. Feed-Forward Network   │
│    ├─ Linear(384 → 1536)  │
│    ├─ GELU()              │
│    └─ Linear(1536 → 384)  │
└───────────────────────────┘
    ↓ + Residual
    ↓ LayerNorm
输出 (B, 197, 384)
```

**Multi-Head Self-Attention:**
```
对于每个token:
1. 计算 Q = x·W_Q, K = x·W_K, V = x·W_V
2. Attention(Q,K,V) = softmax(QK^T/√d_k)·V
3. 让每个patch"看到"所有其他patches
4. 捕捉全局上下文关系
```

### 2.3 Gaze Regression Head

**结构:**
```python
输入: tokens (B, 197, 384)
    ↓ 提取CLS token
(B, 384)
    ↓ LayerNorm
(B, 384)
    ↓ Linear(384 → 256)
(B, 256)
    ↓ GELU激活
(B, 256)
    ↓ Dropout(0.1)
(B, 256)
    ↓ Linear(256 → 2)
(B, 2) ← 最终输出: [x, y] in [-1, 1]
```

**为什么只用CLS token?**
- CLS token在Transformer中聚合了所有信息
- 类似于BERT中的[CLS]
- 包含全局图像表示

---

## 🔄 三、三种运行模式对比

### 模式对比表

| 维度 | 单Exit推理 | 多Exit训练 | 显式Exit选择 |
|------|-----------|-----------|------------|
| **调用方式** | `model(x, state='pursuit')` | `model.forward_all_exits(x)` | `model(x, exit_index=1)` |
| **计算路径** | 只到指定Exit | 计算所有Exit | 只到指定Exit |
| **返回值** | (B, 2) 一个输出 | dict 三个输出 | (B, 2) 一个输出 |
| **使用场景** | 实际部署推理 | 模型训练 | 调试分析 |
| **计算效率** | 高（早停） | 低（全计算） | 高（早停） |
| **状态感知** | ✓ 是 | ✗ 否 | ✗ 否 |

### 模式1: 单Exit推理（最重要）

**流程图:**
```
输入图像 + 状态
    ↓
[查表: state → exit_index]
    ↓
saccade → exit_index=0 → Block 1-3  → Exit 1
pursuit → exit_index=1 → Block 1-6  → Exit 2
fixation → exit_index=2 → Block 1-12 → Exit 3
    ↓
只计算到对应的Block
    ↓
用对应的GazeHead预测
    ↓
返回 (B, 2) 坐标
```

**代码实现:**
```python
def forward(self, x, state='fixation'):
    # 1. 状态映射
    if state == 'saccade':
        exit_idx = 0  # Block 3
    elif state == 'pursuit':
        exit_idx = 1  # Block 6
    else:
        exit_idx = 2  # Block 12
    
    # 2. 前向到指定Block
    exit_block = self.exit_points[exit_idx] - 1  # 转为0-indexed
    tokens = self.forward_features(x, exit_after=exit_block)
    
    # 3. 用对应的Head预测
    gaze = self.exit_heads[f'exit_{exit_idx+1}'](tokens)
    
    return gaze
```

**计算量对比:**
```
Saccade (Exit 1): Block 1-3  → 3/12  = 25%  计算
Pursuit (Exit 2): Block 1-6  → 6/12  = 50%  计算
Fixation (Exit 3): Block 1-12 → 12/12 = 100% 计算
```

### 模式2: 多Exit训练

**流程图:**
```
输入图像
    ↓
┌─────────────────┐
│ 前向到Block 3   │ → Exit 1 → pred_1 (B,2)
│ 前向到Block 6   │ → Exit 2 → pred_2 (B,2)
│ 前向到Block 12  │ → Exit 3 → pred_3 (B,2)
└─────────────────┘
    ↓
返回 {
  'exit_1': pred_1,
  'exit_2': pred_2,
  'exit_3': pred_3
}
```

**Loss计算:**
```python
# 每个Exit计算损失
L1 = Angular_Loss(pred_1, target)
L2 = Angular_Loss(pred_2, target)
L3 = Angular_Loss(pred_3, target)

# 加权组合
L_total = 0.3×L1 + 0.3×L2 + 0.4×L3
```

**为什么这样训练?**
- 浅层Exit学习粗略特征（快速但不精确）
- 深层Exit学习精细特征（慢但精确）
- 所有Exit同时优化，共享底层特征

---

## 🧮 四、损失函数数学原理

### 4.1 Angular Loss

**公式推导:**
```
1. 归一化坐标空间 [-1, 1]
   pred = (x_p, y_p) ∈ [-1,1]²
   target = (x_t, y_t) ∈ [-1,1]²

2. 欧氏距离
   d = √[(x_p - x_t)² + (y_p - y_t)²]

3. 转换为角度（简化）
   θ ≈ d × 90°
   
   原理: 在[-1,1]空间中，距离1对应约90°视角
```

**代码:**
```python
def angular_loss(pred, target):
    diff = pred - target
    distance = torch.norm(diff, dim=1)
    angular_error = distance * 90.0  # 转为度
    return angular_error.mean()
```

### 4.2 Multi-Exit Loss

**数学表达:**
```
L_multi = Σᵢ λᵢ · L(predᵢ, target)

其中:
- i ∈ {1, 2, 3} 表示Exit编号
- λ₁ = 0.3, λ₂ = 0.3, λ₃ = 0.4
- L 可以是 Angular Loss, MSE, 或 L1
```

**为什么λ₃最大?**
```
Exit 1 (浅): 特征粗糙 → 权重小 (0.3)
Exit 2 (中): 特征中等 → 权重中 (0.3)
Exit 3 (深): 特征精细 → 权重大 (0.4)

保证最终模型的深层表现最好
```

### 4.3 State-Aware Loss

**公式:**
```
L_state = (1/N) Σᵢ wₛᵢ · L(predᵢ, targetᵢ)

其中 sᵢ 是样本i的状态:
w_fixation = 1.5  (高精度要求)
w_pursuit = 1.0   (中等精度)
w_saccade = 0.5   (低精度要求)
```

**直觉:**
```
Fixation: 眼睛稳定 → 应该预测很准 → 惩罚更重
Pursuit: 眼睛跟踪 → 中等精度 → 正常惩罚
Saccade: 眼睛快速移动 → 允许误差 → 惩罚更轻
```

---

## 🔬 五、验证方法详解

### 5.1 基础功能验证

**验证项目:**
```
✓ 模型能创建
✓ Exit数量正确(3个)
✓ 状态映射正确
✓ 参数可统计
```

**检查代码:**
```python
model = MultiExitViT(exit_points=[3, 6, 12])
assert len(model.exit_heads) == 3
assert model.state_to_exit == {
    'saccade': 0, 
    'pursuit': 1, 
    'fixation': 2
}
```

### 5.2 推理模式验证

**验证项目:**
```
✓ 每个状态能正常推理
✓ 输出shape正确 (B, 2)
✓ 输出无NaN/Inf
✓ 不同状态使用不同Exit
```

**检查代码:**
```python
model.eval()
with torch.no_grad():
    for state in ['saccade', 'pursuit', 'fixation']:
        gaze = model(image, state=state)
        assert gaze.shape == (B, 2)
        assert not torch.isnan(gaze).any()
```

### 5.3 训练模式验证

**验证项目:**
```
✓ 返回3个Exit的输出
✓ 所有输出shape正确
✓ Loss能计算
✓ 梯度能反向传播
✓ 参数能更新
```

**检查代码:**
```python
model.train()
outputs = model.forward_all_exits(image)
assert len(outputs) == 3

loss, _ = criterion(outputs, target, states)
loss.backward()
optimizer.step()
# 验证参数确实变化了
```

### 5.4 效率验证

**验证项目:**
```
✓ Exit 1参数量 < Exit 2 < Exit 3
✓ 计算节省符合预期
✓ 推理时间差异明显
```

**检查代码:**
```python
params_1 = model.get_num_params(exit_index=0)
params_2 = model.get_num_params(exit_index=1)
params_3 = model.get_num_params(exit_index=2)

assert params_1 < params_2 < params_3
print(f"Exit 1节省: {(1-params_1/params_3)*100:.1f}%")
```

### 5.5 状态路由验证

**验证项目:**
```
✓ 相同输入，不同状态 → 不同输出
✓ Saccade最快 (Exit 1)
✓ Pursuit中等 (Exit 2)
✓ Fixation最慢但最准 (Exit 3)
```

**检查代码:**
```python
with torch.no_grad():
    g1 = model(img, state='saccade')
    g2 = model(img, state='pursuit')
    g3 = model(img, state='fixation')

# 应该有差异
assert not torch.equal(g1, g2)
assert not torch.equal(g2, g3)
```

---

## 🎯 六、完整验证清单

### 运行验证脚本

```bash
python verify_model.py
```

### 预期输出

```
======================================================================
Multi-Exit ViT 完整验证
======================================================================

测试1：模型创建
----------------------------------------------------------------------
✓ 模型创建成功
✓ Exit数量正确: 3个
✓ 状态映射正确

测试2：单Exit推理模式（Inference）
----------------------------------------------------------------------
✓ saccade    -> Exit 1 (Block 3)
✓ pursuit    -> Exit 2 (Block 6)
✓ fixation   -> Exit 3 (Block 12)
✓ 单Exit推理模式工作正常

测试3：多Exit训练模式（Training）
----------------------------------------------------------------------
✓ 返回了3个Exit的输出
  exit_1: torch.Size([4, 2]) ✓
  exit_2: torch.Size([4, 2]) ✓
  exit_3: torch.Size([4, 2]) ✓
✓ 多Exit训练模式工作正常

测试4：验证计算效率
----------------------------------------------------------------------
  Exit 1 (Block 3): 21,123,456 params
  Exit 2 (Block 6): 21,456,789 params
  Exit 3 (Block 12): 22,012,345 params
✓ Exit 1比Exit 3节省 4.0% 参数

测试5：损失函数
----------------------------------------------------------------------
✓ Angular Loss: 45.2341°
✓ Multi-Exit Loss: 38.1234
✓ State-Aware Loss: 42.5678

测试6：反向传播与参数更新
----------------------------------------------------------------------
✓ 反向传播成功
✓ 参数已更新

测试7：状态路由正确性
----------------------------------------------------------------------
✓ 状态路由工作正常（不同状态产生不同输出）

======================================================================
✓✓✓ 所有测试通过！Multi-Exit ViT模型实现正确 ✓✓✓
======================================================================
```

---

## 📝 七、常见问题解答

### Q1: 为什么不是所有状态都用最深的Exit?

**A:** 计算效率与精度的权衡
- Saccade: 眼睛快速移动，不需要很高精度，用Exit 1节省66%计算
- Fixation: 眼睛稳定注视，需要高精度，用Exit 3保证质量
- Pursuit: 介于两者之间

### Q2: 训练时为什么要计算所有Exit?

**A:** 让每个Exit都学到有用的特征
- 如果只训练Exit 3，Exit 1和2的head是随机的
- 同时训练让浅层Exit也能做出合理预测
- 提供了不同深度的特征表示

### Q3: 如何选择Exit点?

**A:** 经验法则
- 早期 (1/4位置): Block 3  → 粗略特征
- 中期 (1/2位置): Block 6  → 中等特征  
- 晚期 (全部): Block 12 → 完整特征

### Q4: Angular Loss的90度系数从哪来?

**A:** 简化的屏幕-角度映射
- [-1, 1] 坐标空间对应屏幕全范围
- 人眼视场约±45°
- 距离1 ≈ 90°视角（简化）
- 实际可根据具体屏幕几何校准

---

## 🎓 八、总结

### 核心要点

1. **Multi-Exit架构**: 3个Exit在不同深度提供输出
2. **状态感知**: 根据眼动状态智能选择Exit
3. **训练策略**: 所有Exit同时优化，加权组合
4. **推理效率**: 早期Exit可节省66%计算

### 验证要点

1. **功能完整性**: 所有模式都能工作
2. **数值正确性**: 无NaN/Inf，shape正确
3. **梯度流动**: 能反向传播和更新
4. **效率差异**: Exit1 < Exit2 < Exit3

### 下一步

运行完整验证:
```bash
python verify_model.py
```

如果所有测试通过，就可以开始训练了！
