# Python 环境配置指南

## 当前状态

❌ 系统中没有可用的Python环境
✅ 所有代码和数据已就绪
✅ 需要安装Python和依赖包

---

## 方案1：安装Anaconda（推荐）⭐

### 为什么推荐Anaconda？
- 包含Python和常用科学计算包
- 环境管理方便
- 适合深度学习项目

### 安装步骤

1. **下载Anaconda**
   - 访问：https://www.anaconda.com/download
   - 选择：Windows版本
   - 下载：Anaconda3-2024.xx-Windows-x86_64.exe

2. **安装Anaconda**
   - 双击安装包
   - 选择：Just Me (recommended)
   - **重要**：勾选 "Add Anaconda to my PATH environment variable"
   - 安装位置：默认即可（或选择 C:\Anaconda3）
   - 点击：Install

3. **验证安装**
   打开新的命令行窗口：
   ```bash
   conda --version
   python --version
   ```

4. **创建项目环境**
   ```bash
   cd C:\workspace\fangzeyu\prj\event_based_gaze_tracking
   conda create -n gaze_tracking python=3.9 -y
   conda activate gaze_tracking
   ```

5. **安装依赖**
   ```bash
   pip install torch torchvision timm --index-url https://download.pytorch.org/whl/cu118
   pip install opencv-python numpy tqdm tensorboard
   ```

6. **运行验证**
   ```bash
   python verify_model.py
   ```

---

## 方案2：安装官方Python

### 安装步骤

1. **下载Python**
   - 访问：https://www.python.org/downloads/
   - 下载：Python 3.9.x 或 3.10.x (Windows installer 64-bit)

2. **安装Python**
   - 双击安装包
   - **重要**：勾选 "Add Python to PATH"
   - 点击："Install Now"

3. **验证安装**
   打开新的命令行窗口：
   ```bash
   python --version
   pip --version
   ```

4. **安装依赖**
   ```bash
   cd C:\workspace\fangzeyu\prj\event_based_gaze_tracking
   pip install torch torchvision timm --index-url https://download.pytorch.org/whl/cu118
   pip install opencv-python numpy tqdm tensorboard
   ```

5. **运行验证**
   ```bash
   python verify_model.py
   ```

---

## 方案3：使用便携版Python（快速方案）

如果不想安装，可以下载便携版：

1. **下载WinPython**
   - 访问：https://winpython.github.io/
   - 下载：WinPython 3.9.x 64bit

2. **解压并使用**
   ```bash
   # 解压到 C:\WinPython
   # 进入目录
   cd C:\WinPython\WPy64-3920\scripts
   
   # 运行Python
   .\python.exe --version
   ```

---

## 快速诊断

如果安装后Python仍然无法运行，尝试：

### 检查1：PATH环境变量
```bash
echo %PATH%
# 应该包含Python的安装路径
```

### 检查2：重启命令行
安装后必须重启命令行窗口，让PATH生效

### 检查3：使用完整路径
```bash
# 如果Python安装在 C:\Python39
C:\Python39\python.exe --version
```

---

## 安装完成后的验证清单

### ✓ 检查项目

1. [ ] Python已安装
   ```bash
   python --version
   # 输出: Python 3.9.x 或 3.10.x
   ```

2. [ ] pip可用
   ```bash
   pip --version
   # 输出: pip 23.x.x from ...
   ```

3. [ ] PyTorch已安装
   ```bash
   python -c "import torch; print(torch.__version__)"
   # 输出: 2.0.0+cu118
   ```

4. [ ] timm已安装
   ```bash
   python -c "import timm; print(timm.__version__)"
   # 输出: 0.9.x
   ```

5. [ ] 验证脚本通过
   ```bash
   python verify_model.py
   # 输出: ✓✓✓ 所有测试通过！
   ```

---

## 推荐时间线

### 现在（Day 3晚上）
- 安装Anaconda或Python（15-30分钟）
- 安装依赖包（10-15分钟）
- 运行 verify_model.py 验证（2分钟）

### 明天（Day 4）
- 生成Mock数据
- 开始训练

---

## 常见问题

### Q: 安装需要多长时间？
A: 
- Anaconda：下载15-20分钟，安装5分钟
- 官方Python：下载5分钟，安装2分钟
- 依赖包：10-15分钟

### Q: 需要多少磁盘空间？
A: 
- Anaconda：约5GB
- Python + 依赖：约3GB

### Q: 我有多个Python版本怎么办？
A: 使用Anaconda的虚拟环境可以完美隔离

### Q: 安装过程中出错怎么办？
A: 
1. 确保以管理员权限运行
2. 关闭杀毒软件
3. 检查磁盘空间
4. 查看错误信息并搜索解决方案

---

## 联系方式

安装完成后，运行：
```bash
python verify_model.py
```

如果看到：
```
✓✓✓ 所有测试通过！Multi-Exit ViT模型实现正确 ✓✓✓
```

就说明环境配置成功，可以开始训练了！

---

## 备注

- 推荐使用 **Anaconda**，最省心
- Python版本推荐 **3.9** 或 **3.10**
- 确保勾选 **"Add to PATH"** 选项
- 安装后 **重启命令行窗口**
