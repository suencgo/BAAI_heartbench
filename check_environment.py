#!/usr/bin/env python3
"""
检查 Heart Bench 环境是否正确配置
"""

import sys

def check_environment():
    """检查环境配置"""
    print("=" * 60)
    print("Heart Bench 环境检查")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # 检查 Python 版本
    print("\n[1] 检查 Python 版本...")
    python_version = sys.version_info
    print(f"    Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        errors.append("Python 版本需要 >= 3.8")
    else:
        print("    [OK] Python 版本符合要求")
    
    # 检查必需的包
    print("\n[2] 检查依赖包...")
    required_packages = {
        'openai': 'openai',
        'dashscope': 'dashscope',
        'requests': 'requests',
        'tqdm': 'tqdm',
        'PIL': 'Pillow'
    }
    
    optional_packages = {
        'qwen_vl_utils': 'qwen-vl-utils (可选，仅在使用 Qwen 模型时需要)'
    }
    
    missing_packages = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"    [OK] {package_name} 已安装")
        except ImportError:
            missing_packages.append(package_name)
            print(f"    [FAIL] {package_name} 未安装")
    
    # 检查可选包
    for module_name, package_name in optional_packages.items():
        try:
            __import__(module_name)
            print(f"    [OK] {package_name} 已安装")
        except ImportError:
            print(f"    [WARN] {package_name} 未安装（可选）")
    
    if missing_packages:
        errors.append(f"缺少以下包: {', '.join(missing_packages)}")
        print(f"\n    安装命令: pip install {' '.join(missing_packages)}")
    
    # 检查项目文件
    print("\n[3] 检查项目文件...")
    from pathlib import Path
    
    required_files = [
        'prompt_manager.py',
        'model_apis.py',
        'evaluate_benchmark.py',
        'config_manager.py',
        'answer_parser.py',
        'requirements.txt'
    ]
    
    project_root = Path(__file__).parent
    missing_files = []
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"    [OK] {file_name} 存在")
        else:
            missing_files.append(file_name)
            print(f"    [FAIL] {file_name} 不存在")
    
    if missing_files:
        warnings.append(f"缺少以下文件: {', '.join(missing_files)}")
    
    # 检查配置文件
    print("\n[4] 检查配置文件...")
    config_file = project_root / 'model_config.json'
    if config_file.exists():
        print("    [OK] model_config.json 存在")
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            if 'models' in config and len(config['models']) > 0:
                print(f"    [OK] 配置文件包含 {len(config['models'])} 个模型配置")
            else:
                warnings.append("配置文件为空或格式不正确")
        except Exception as e:
            warnings.append(f"配置文件读取失败: {e}")
    else:
        warnings.append("model_config.json 不存在，需要创建配置文件")
        print("    [WARN] model_config.json 不存在（可选）")
    
    # 总结
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    
    if errors:
        print("\n[ERROR] 发现错误:")
        for error in errors:
            print(f"  - {error}")
        print("\n请修复上述错误后重试。")
        return False
    else:
        print("\n[OK] 所有必需组件检查通过！")
    
    if warnings:
        print("\n[WARN] 警告:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\n这些警告不会影响基本功能，但建议处理。")
    
    print("\n环境检查完成！")
    return True

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
