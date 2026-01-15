#!/bin/bash
# 使用 venv 创建环境（快速方案）

set -e

echo "=========================================="
echo "Heart Bench 环境配置 (使用 venv)"
echo "=========================================="
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] 未找到 python3，请先安装 Python"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "[INFO] 使用 $PYTHON_VERSION"

# 创建虚拟环境
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "[WARN] 虚拟环境已存在: $VENV_DIR"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "使用现有环境"
        source "$VENV_DIR/bin/activate"
        pip install -r requirements.txt
        echo ""
        echo "[OK] 环境已激活，依赖已安装"
        echo "使用 'source venv/bin/activate' 激活环境"
        exit 0
    fi
fi

echo "创建虚拟环境: $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "激活虚拟环境..."
source "$VENV_DIR/bin/activate"

echo "升级 pip..."
pip install --upgrade pip

echo "安装依赖..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "[OK] 环境创建完成！"
echo "=========================================="
echo ""
echo "激活环境："
echo "  source venv/bin/activate"
echo ""
echo "退出环境："
echo "  deactivate"
echo ""
