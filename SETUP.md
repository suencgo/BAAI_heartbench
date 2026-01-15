# Heart Bench 环境设置指南

本指南将帮助您使用 conda 创建和管理 Heart Bench 的运行环境。

## 快速开始

### 1. 创建 Conda 环境

```bash
# 进入项目目录
cd Heart_bench

# 使用 environment.yml 创建环境（推荐）
conda env create -f environment.yml

# 或者手动创建
conda create -n heart_bench python=3.10 -y
conda activate heart_bench
pip install -r requirements.txt
```

### 2. 激活环境

```bash
conda activate heart_bench
```

### 3. 验证安装

```bash
# 检查 Python 版本
python --version  # 应该显示 Python 3.10.x

# 检查依赖包
python -c "import openai, dashscope, qwen_vl_utils, requests, tqdm, PIL; print('All dependencies installed!')"
```

## 详细步骤

### 方法一：使用 environment.yml（推荐）

这是最简单的方法，会自动安装所有依赖：

```bash
cd Heart_bench
conda env create -f environment.yml
conda activate heart_bench
```

### 方法二：手动创建

如果您想自定义 Python 版本或其他设置：

```bash
# 创建环境（可以指定不同的 Python 版本）
conda create -n heart_bench python=3.10 -y

# 激活环境
conda activate heart_bench

# 安装依赖
pip install -r requirements.txt
```

### 方法三：从现有环境克隆

如果您想基于现有环境创建：

```bash
# 克隆现有环境
conda create --name heart_bench --clone base

# 激活环境
conda activate heart_bench

# 安装依赖
pip install -r requirements.txt
```

## 环境管理

### 激活和退出

```bash
# 激活环境
conda activate heart_bench

# 退出环境
conda deactivate
```

### 查看环境信息

```bash
# 查看所有 conda 环境
conda env list

# 查看当前环境的包列表
conda list

# 查看环境详细信息
conda info --envs
```

### 更新依赖

```bash
# 激活环境
conda activate heart_bench

# 更新 requirements.txt 中的包
pip install --upgrade -r requirements.txt
```

### 删除环境

如果不再需要该环境：

```bash
# 先退出环境
conda deactivate

# 删除环境
conda env remove -n heart_bench
```

## 常见问题

### 1. Conda 命令未找到

如果提示 `conda: command not found`，需要先初始化 conda：

```bash
# 对于 bash
conda init bash
source ~/.bashrc

# 对于 zsh
conda init zsh
source ~/.zshrc
```

### 2. 环境激活失败

如果激活环境时出现问题，尝试：

```bash
# 使用完整路径
source ~/anaconda3/etc/profile.d/conda.sh  # 或 miniconda3
conda activate heart_bench
```

### 3. 依赖安装失败

如果某些包安装失败：

```bash
# 先更新 pip
pip install --upgrade pip

# 再安装依赖
pip install -r requirements.txt

# 如果还有问题，可以逐个安装
pip install openai>=1.0.0
pip install dashscope>=1.14.0
# ... 等等
```

### 4. Python 版本不兼容

如果遇到 Python 版本问题：

```bash
# 创建指定版本的环境
conda create -n heart_bench python=3.9 -y  # 或 3.10, 3.11 等
conda activate heart_bench
pip install -r requirements.txt
```

## 测试环境

创建环境后，可以运行测试脚本验证：

```bash
# 激活环境
conda activate heart_bench

# 运行测试（需要先配置 model_config.json）
python test_single_patient.py --patient_id 1322705 --model_alias qwen3-vl-235b
```

## 最佳实践

1. **每次使用前激活环境**：确保在正确的环境中运行代码
2. **定期更新依赖**：保持包的最新版本以获得 bug 修复和新功能
3. **使用 environment.yml**：方便在不同机器上重建相同的环境
4. **不要混用 conda 和 pip**：尽量使用一种包管理器，避免冲突

## 导出环境配置

如果您修改了环境并想保存配置：

```bash
# 激活环境
conda activate heart_bench

# 导出环境配置
conda env export > environment.yml

# 或者只导出 pip 包
pip freeze > requirements.txt
```
