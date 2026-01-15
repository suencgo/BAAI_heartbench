# Heart Bench 快速开始指南

## 环境已配置完成！

您的 Heart Bench 环境已经成功创建并配置。

## 激活环境

每次使用前，请先激活虚拟环境：

```bash
cd Heart_bench
source venv/bin/activate
```

激活后，您会看到命令提示符前面有 `(venv)` 标识。

## 验证环境

运行环境检查脚本：

```bash
python check_environment.py
```

## 使用示例

### 1. 测试单个病人

```bash
# 激活环境
source venv/bin/activate

# 运行测试
python test_single_patient.py --patient_id 1322705 --model_alias qwen3-vl-235b
```

### 2. 评估基准测试

```bash
# 激活环境
source venv/bin/activate

# 运行评估
python evaluate_benchmark.py \
    --json_path ../dataset/patient_1322705_vqa_png.json \
    --image_base_dir ../dataset \
    --test_model_alias qwen3-vl-235b \
    --max_workers 4
```

### 3. 批量测试

```bash
# 激活环境
source venv/bin/activate

# 批量测试所有病人的特定序列
python batch_test_cine.py --model_alias qwen3-vl-235b --filter_sequence cine
```

## 退出环境

使用完毕后，退出虚拟环境：

```bash
deactivate
```

## 注意事项

1. **每次使用前都要激活环境**：确保在正确的环境中运行代码
2. **qwen-vl-utils 警告**：如果看到 qwen-vl-utils 未安装的警告，这是正常的。该包是可选的，仅在直接使用 Qwen 模型时需要。如果使用 Ksyun API（兼容 OpenAI 格式），则不需要。
3. **配置文件**：确保 `model_config.json` 中已配置好您的 API 密钥

## 常见问题

### 环境激活失败

如果 `source venv/bin/activate` 失败，尝试：

```bash
. venv/bin/activate
```

### 需要重新安装依赖

如果依赖有问题，可以重新安装：

```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### 删除环境重新创建

```bash
rm -rf venv
./setup_venv.sh
```

## 下一步

- 查看 [README.md](README.md) 了解完整功能
- 查看 [SETUP.md](SETUP.md) 了解详细的环境配置说明
- 配置 `model_config.json` 添加您的 API 密钥
