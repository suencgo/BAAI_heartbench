# 快速开始指南

## 文件结构

```
Heart_bench/
├── __init__.py              # 包初始化文件
├── prompt_manager.py        # Prompt管理器（测试模型和Judge模型的prompt生成）
├── model_apis.py           # 模型API接口（GPT、Qwen、Ksyun）
├── config_manager.py       # 配置管理器
├── answer_parser.py        # 答案解析器
├── evaluate_benchmark.py   # 主评估脚本
├── example_usage.py        # 使用示例
├── example_config.py       # 配置使用示例
├── model_config.json       # 模型配置文件
├── requirements.txt        # 依赖包
├── README.md              # 详细文档
├── QUICKSTART.md          # 快速开始指南
└── .gitignore            # Git忽略文件
```

## 快速开始

### 1. 安装依赖

```bash
cd Heart_bench
pip install -r requirements.txt
```

### 2. 配置模型（推荐方式）

编辑 `model_config.json` 文件，配置你的模型。例如，`qwen3-vl-235b` 别名对应实际的 `qwen3-vl-235b-a22b-thinking` 模型：

```json
{
  "models": {
    "qwen3-vl-235b": {
      "type": "ksyun",
      "model": "qwen3-vl-235b-a22b-thinking",
      "api_key": "your-api-key",
      "base_url": "https://kspmas.ksyun.com/v1/",
      "max_tokens": 500,
      "temperature": 0.0,
      "description": "Kingsoft Cloud Qwen3 VL 235B Model"
    },
    "gpt-4o": {
      "type": "gpt",
      "model": "gpt-4o",
      "api_key": "your-api-key",
      "base_url": null,
      "max_tokens": 500,
      "temperature": 0.0,
      "description": "OpenAI GPT-4o Model"
    }
  },
  "default_model": "qwen3-vl-235b",
  "default_judge_model": "qwen3-vl-235b"
}
```

**注意**：请将 `your-api-key` 替换为你的实际API密钥。

或者使用环境变量（传统方式）：

```bash
# GPT模型
export OPENAI_API_KEY="your-openai-api-key"

# Qwen模型
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

### 3. 运行评估

#### 使用模型别名（推荐，更直观）

```bash
cd Heart_bench
python evaluate_benchmark.py \
    --json_path ../1322705/patient_1322705_vqa_png.json \
    --image_base_dir ../1322705 \
    --test_model_alias qwen3-vl-235b
```

结果会自动保存到 `Heart_bench/output/qwen3-vl-235b/` 目录。

#### 使用传统方式（GPT模型）

```bash
cd Heart_bench
python evaluate_benchmark.py \
    --json_path ../1322705/patient_1322705_vqa_png.json \
    --image_base_dir ../1322705 \
    --test_model_type gpt \
    --test_model_name gpt-4o \
    --test_api_key $OPENAI_API_KEY
```

#### 使用传统方式（Qwen模型）

```bash
cd Heart_bench
python evaluate_benchmark.py \
    --json_path ../1322705/patient_1322705_vqa_png.json \
    --image_base_dir ../1322705 \
    --test_model_type qwen \
    --test_model_name qwen-vl-max \
    --test_api_key $DASHSCOPE_API_KEY
```

#### 指定输出格式

```bash
python evaluate_benchmark.py \
    --json_path ../1322705/patient_1322705_vqa_png.json \
    --image_base_dir ../1322705 \
    --test_model_alias qwen3-vl-235b \
    --output_format both  # json, tsv, 或 both
```

#### 使用Judge模型评估

```bash
python evaluate_benchmark.py \
    --json_path ../1322705/patient_1322705_vqa_png.json \
    --image_base_dir ../1322705 \
    --test_model_alias qwen3-vl-235b \
    --use_judge \
    --judge_model_alias qwen3-vl-235b
```

## 主要功能

1. **多模型支持**：支持GPT、Qwen、Ksyun等不同API模型
2. **配置管理**：通过配置文件管理模型别名、API密钥等，使用更直观
3. **智能Prompt**：针对不同序列类型（cine_sax, LGE_sax, perfusion等）和题目类型（单选/多选）自动生成优化的prompt
4. **灵活评估**：
   - 规则评估：基于答案提取和匹配的快速评估
   - Judge模型评估：使用LLM进行更智能的语义评估
5. **详细报告**：生成JSON和TSV格式的详细评估报告，包含答案推理原因
6. **评估指标**：支持准确率（accuracy）和多选题命中率（hit）指标

## 支持的序列类型

- `cine_sax`: Cine序列短轴切面
- `cine_4ch`: Cine序列四腔心切面
- `cine_3ch`: Cine序列三腔心切面
- `LGE_sax`: LGE序列短轴切面
- `LGE_4ch`: LGE序列四腔心切面
- `perfusion`: 灌注序列
- `T2_sax`: T2序列短轴切面

## 输出文件

评估完成后，会在输出目录（默认：`Heart_bench/output/{model_name}/`）生成：

- `detailed_results.json`: 每个问题的详细评估结果，包含：
  - `qid`, `field`, `patient_id`, `question_type`, `sequence_view`, `original_nii`
  - `gt`: 标准答案
  - `answer`: 模型答案
  - `reason`: 模型推理原因
  - `is_correct`: 是否正确
  - `accuracy`: 准确率
  - `hit`: 命中率（仅多选题）
- `summary.json`: 总体统计摘要（按字段、序列视图、问题类型）
- `report.txt`: 可读的文本报告
- `detailed_results.tsv`: TSV格式的详细结果（如果指定了 `--output_format tsv` 或 `both`）

## 代码示例

### 使用模型别名（推荐）

```python
from evaluate_benchmark import BenchmarkDataLoader, BenchmarkEvaluator
from model_apis import ModelFactory
from config_manager import ModelConfigManager

# 初始化配置管理器
config_manager = ModelConfigManager()

# 使用模型别名创建模型（更直观）
test_model = ModelFactory.create_model(
    model_alias='qwen3-vl-235b',  # 使用别名
    config_manager=config_manager
)

# 加载数据
data_loader = BenchmarkDataLoader(
    json_path='../1322705/patient_1322705_vqa_png.json',
    image_base_dir='../1322705'
)

# 创建评估器
evaluator = BenchmarkEvaluator(
    data_loader, 
    test_model,
    include_reason=True  # 要求模型输出推理原因
)

# 执行评估
result = evaluator.evaluate()

# 保存结果（会自动保存到 output/{model_name}/ 目录）
evaluator.save_results(result, output_format='both')

# 打印结果
print(f"准确率: {result.accuracy:.4f}")
```

### 传统方式（直接指定模型类型）

```python
from evaluate_benchmark import BenchmarkDataLoader, BenchmarkEvaluator
from model_apis import ModelFactory
import os

# 创建模型
test_model = ModelFactory.create_model(
    model_type='gpt',
    api_key=os.getenv('OPENAI_API_KEY'),
    model='gpt-4o'
)

# 后续步骤相同...
```

### 使用Judge模型

```python
from evaluate_benchmark import BenchmarkDataLoader, BenchmarkEvaluator
from model_apis import ModelFactory
from config_manager import ModelConfigManager

config_manager = ModelConfigManager()

# 创建测试模型和Judge模型
test_model = ModelFactory.create_model(
    model_alias='qwen3-vl-235b',
    config_manager=config_manager
)

judge_model = ModelFactory.create_model(
    model_alias='qwen3-vl-235b',
    config_manager=config_manager
)

# 创建评估器（使用Judge模型）
evaluator = BenchmarkEvaluator(
    data_loader, 
    test_model,
    judge_model=judge_model,
    use_judge=True
)

# 执行评估
result = evaluator.evaluate()
```

## 评估指标

### 单选题
- **accuracy**: 准确率（完全匹配为1.0，否则为0.0）

### 多选题
- **accuracy**: 准确率（所有答案完全匹配为1.0，否则为0.0）
- **hit**: 命中率（至少选中一个正确答案为1.0，否则为0.0）

## 注意事项

1. **图片路径**：确保JSON中的图片路径正确，且相对于`image_base_dir`
2. **API配额**：评估大量题目会消耗API配额，建议先用少量题目测试
3. **多图片输入**：每个问题可能包含多张图片（多个slice），模型需要支持多图输入
4. **Judge模型**：使用Judge模型会增加API调用成本，但评估更准确
5. **输出目录**：如果不指定`--output_dir`，结果会自动保存到`Heart_bench/output/{model_name}/`目录
6. **API密钥安全**：不要在配置文件中提交真实的API密钥，建议使用环境变量或`.gitignore`忽略配置文件

## 更多信息

- 详细文档请参考 [README.md](README.md)
- 配置管理示例请参考 [example_config.py](example_config.py)
- 使用示例请参考 [example_usage.py](example_usage.py)
