# Miniconda 快速安装指南（推荐）

## 为什么选择Miniconda？
- ✅ 无需注册，直接下载
- ✅ 体积小（约100MB vs Anaconda的500MB）
- ✅ 安装快（2分钟 vs 10分钟）
- ✅ 功能完全够用

---

## 📥 下载

### 直接下载链接（无需注册）
```
https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
```

或者从清华镜像下载（更快）：
```
https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe
```

---

## 🔧 安装步骤

### 1. 双击安装包

### 2. 安装选项
- 选择：Just Me (recommended)
- 安装位置：默认 (C:\Users\你的用户名\miniconda3)
- **重要**：勾选这两项
  - ☑ Add Miniconda3 to my PATH environment variable
  - ☑ Register Miniconda3 as my default Python

### 3. 点击 Install

### 4. 安装完成后，重启命令行

---

## ✅ 验证安装

打开**新的**命令行窗口，输入：
```bash
conda --version
python --version
```

应该看到：
```
conda 24.x.x
Python 3.11.x
```

---

## 🚀 配置项目环境

### 1. 进入项目目录
```bash
cd C:\workspace\fangzeyu\prj\event_based_gaze_tracking
```

### 2. 创建虚拟环境
```bash
conda create -n gaze_tracking python=3.9 -y
```

### 3. 激活环境
```bash
conda activate gaze_tracking
```

### 4. 安装PyTorch（使用清华镜像，更快）
```bash
pip install torch torchvision timm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5. 安装其他依赖
```bash
pip install opencv-python numpy tqdm tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 🧪 运行验证

```bash
python verify_model.py
```

预期输出：
```
✓✓✓ 所有测试通过！Multi-Exit ViT模型实现正确 ✓✓✓
```

---

## ⏱️ 时间估算

- 下载Miniconda: 2-5分钟（约100MB）
- 安装Miniconda: 2分钟
- 创建环境: 1分钟
- 安装PyTorch: 5-10分钟
- 安装其他包: 2-3分钟
- **总计**: 15-20分钟

---

## 🔄 如果下载速度慢

### 使用清华镜像（国内速度快）

#### Miniconda下载：
```
https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe
```

#### 配置conda镜像：
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

#### 配置pip镜像：
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 🆘 常见问题

### Q1: 安装后找不到conda命令？
**A**: 重启命令行窗口，或者重启电脑

### Q2: conda activate不工作？
**A**: 运行：
```bash
conda init
```
然后重启命令行

### Q3: pip install很慢？
**A**: 使用清华镜像：
```bash
pip install 包名 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q4: 提示权限不足？
**A**: 右键"命令提示符" → 以管理员身份运行

---

## 📝 完整命令清单（复制粘贴即可）

```bash
# 1. 验证安装
conda --version
python --version

# 2. 进入项目
cd C:\workspace\fangzeyu\prj\event_based_gaze_tracking

# 3. 创建环境
conda create -n gaze_tracking python=3.9 -y

# 4. 激活环境
conda activate gaze_tracking

# 5. 安装依赖（使用清华镜像）
pip install torch torchvision timm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python numpy tqdm tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple

# 6. 验证
python verify_model.py

# 7. 如果验证通过，生成Mock数据
python src/data/mock_dataset.py

# 8. 开始训练
python src/train/train_multi_exit_vit.py
```

---

## ✨ 优势总结

使用Miniconda的优势：
- ✅ 无需注册账号
- ✅ 下载快（100MB vs 500MB）
- ✅ 安装快（2分钟 vs 10分钟）
- ✅ 功能完全够用
- ✅ 环境管理方便

---

## 🎯 下一步

1. 下载Miniconda（2-5分钟）
2. 安装（2分钟）
3. 配置环境（10分钟）
4. 运行验证
5. 开始训练！

加油！🚀
