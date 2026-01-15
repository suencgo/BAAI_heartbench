#!/bin/bash
# 安装和配置 Miniconda 脚本

set -e

echo "=========================================="
echo "Heart Bench 环境配置脚本"
echo "=========================================="
echo ""

# 检查是否已安装 conda
if command -v conda &> /dev/null; then
    echo "[OK] Conda 已安装"
    conda --version
else
    echo "[INFO] 未检测到 conda，开始安装 Miniconda..."
    
    # 检测系统架构
    ARCH=$(uname -m)
    if [ "$ARCH" == "arm64" ] || [ "$ARCH" == "aarch64" ]; then
        ARCH_TYPE="arm64"
    else
        ARCH_TYPE="x86_64"
    fi
    
    echo "检测到系统架构: $ARCH_TYPE"
    
    # Miniconda 下载 URL
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-${ARCH_TYPE}.sh"
    INSTALLER_PATH="$HOME/miniconda_installer.sh"
    
    echo "下载 Miniconda..."
    curl -L -o "$INSTALLER_PATH" "$MINICONDA_URL"
    
    echo "安装 Miniconda..."
    bash "$INSTALLER_PATH" -b -p "$HOME/miniconda3"
    
    # 初始化 conda
    echo "初始化 conda..."
    "$HOME/miniconda3/bin/conda" init zsh
    
    # 清理安装文件
    rm "$INSTALLER_PATH"
    
    echo ""
    echo "[OK] Miniconda 安装完成！"
    echo ""
    echo "请执行以下命令完成配置："
    echo "  source ~/.zshrc"
    echo "  或者重新打开终端"
    echo ""
    echo "然后运行："
    echo "  conda env create -f environment.yml"
    echo "  conda activate heart_bench"
fi
