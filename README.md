# 医学影像VQA Benchmark评估框架

支持GPT、Qwen和Ksyun等不同API模型的医学影像视觉问答评估框架。

## 功能特点

1. **多模型支持**：支持GPT、Qwen、Ksyun等不同API模型
2. **配置管理**：通过配置文件管理模型别名、API密钥等，使用更直观
3. **智能Prompt**：针对不同序列类型和题目类型自动生成优化的prompt
   - **Cine序列**：11个字段的专门prompt templates（Thickening、Wall Motion、Systolic Function、Valves、Special Signs、Effusion等）
   - **LGE序列**：9个字段的专门prompt templates（Enhancement Status、Abnormal Signal、High Signal分布、Low Signal分布、Special Description等）
   - **Perfusion序列**：5个字段的专门prompt templates（Perfusion Status、Abnormal Segments、Abnormal Regions、Signal Characteristics、Myocardial Layer）
   - **T2序列**：4个字段的专门prompt templates（T2 Signal、Abnormal Segments/Regions、Signal Distribution）
4. **结构化Reason分析**：为每个题目类型提供5步分析框架，引导模型进行深入、有针对性的图像分析
5. **v2设计特性**：
   - 严格的两行输出格式（Line 1: Answer, Line 2: Reason）
   - 多选题包含Z. None选项，避免强制选择
   - 图像仅推理（移除临床解释，保持严格视觉推理）
   - 操作性的视觉定义（特别是灌注的Reduced/Delayed/Defect）
6. **序列过滤**：支持通过 `--filter_sequence` 参数过滤特定序列类型的题目（如只测试cine序列）
7. **批量测试**：提供 `batch_test_cine.py` 脚本，可自动批量测试所有病人的特定序列
8. **灵活评估**：支持规则评估和Judge模型评估两种方式
9. **详细报告**：生成JSON和TSV格式的详细评估报告，包含答案推理原因
10. **评估指标**：支持准确率（accuracy）和多选题命中率（hit）指标

## 文件结构

```
Heart_bench/
├── __init__.py              # 包初始化文件
├── prompt_manager.py        # Prompt管理器（测试模型和Judge模型的prompt生成）
├── model_apis.py           # 模型API接口（GPT、Qwen、Ksyun）
├── config_manager.py       # 配置管理器
├── answer_parser.py        # 答案解析器
├── evaluate_benchmark.py   # 主评估脚本
├── batch_test_cine.py     # 批量测试脚本（用于批量测试所有病人的特定序列）
├── example_usage.py        # 使用示例
├── example_config.py       # 配置使用示例
├── model_config.json       # 模型配置文件
├── requirements.txt        # 依赖包
├── README.md              # 详细文档
├── QUICKSTART.md          # 快速开始指南
└── .gitignore            # Git忽略文件
```

## 安装依赖

```bash
cd Heart_bench
pip install -r requirements.txt
```

## 配置模型

### 方式1：使用配置文件（推荐）

编辑 `model_config.json` 文件，配置你的模型：

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

### 方式2：使用环境变量（传统方式）

```bash
# GPT模型
export OPENAI_API_KEY="your-openai-api-key"

# Qwen模型
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

## 使用方法

### 使用模型别名（推荐）

```bash
cd Heart_bench
python evaluate_benchmark.py \
    --json_path ../1322705/patient_1322705_vqa_png.json \
    --image_base_dir ../1322705 \
    --test_model_alias qwen3-vl-235b
```

### 使用传统方式（GPT模型）

```bash
cd Heart_bench
python evaluate_benchmark.py \
    --json_path ../1322705/patient_1322705_vqa_png.json \
    --image_base_dir ../1322705 \
    --test_model_type gpt \
    --test_model_name gpt-4o \
    --test_api_key $OPENAI_API_KEY
```

### 使用传统方式（Qwen模型）

```bash
cd Heart_bench
python evaluate_benchmark.py \
    --json_path ../1322705/patient_1322705_vqa_png.json \
    --image_base_dir ../1322705 \
    --test_model_type qwen \
    --test_model_name qwen-vl-max \
    --test_api_key $DASHSCOPE_API_KEY
```

### 使用Judge模型评估

```bash
cd Heart_bench
python evaluate_benchmark.py \
    --json_path ../1322705/patient_1322705_vqa_png.json \
    --image_base_dir ../1322705 \
    --test_model_alias qwen3-vl-235b \
    --use_judge \
    --judge_model_alias qwen3-vl-235b
```

### 指定输出格式

```bash
python evaluate_benchmark.py \
    --json_path ../1322705/patient_1322705_vqa_png.json \
    --image_base_dir ../1322705 \
    --test_model_alias qwen3-vl-235b \
    --output_format both  # json, tsv, 或 both
```

### 过滤特定序列类型

使用 `--filter_sequence` 参数可以只测试特定序列类型的题目：

```bash
# 只测试 cine 序列（cine_sax, cine_4ch, cine_3ch 等）
python evaluate_benchmark.py \
    --json_path ../dataset/patient_1322705_vqa_png.json \
    --image_base_dir ../dataset \
    --test_model_alias qwen3-vl-235b \
    --filter_sequence cine

# 只测试 LGE 序列
python evaluate_benchmark.py \
    --json_path ../dataset/patient_1322705_vqa_png.json \
    --image_base_dir ../dataset \
    --test_model_alias qwen3-vl-235b \
    --filter_sequence LGE

# 只测试 perfusion 序列
python evaluate_benchmark.py \
    --json_path ../dataset/patient_1322705_vqa_png.json \
    --image_base_dir ../dataset \
    --test_model_alias qwen3-vl-235b \
    --filter_sequence perfusion

# 只测试 T2 序列
python evaluate_benchmark.py \
    --json_path ../dataset/patient_1322705_vqa_png.json \
    --image_base_dir ../dataset \
    --test_model_alias qwen3-vl-235b \
    --filter_sequence T2
```

### 批量测试所有病人的特定序列（推荐）

使用 `batch_test_cine.py` 脚本可以自动批量测试所有病人的特定序列：

```bash
# 批量测试所有病人的 cine 序列
cd Heart_bench
python batch_test_cine.py --model_alias qwen3-vl-235b --filter_sequence cine

# 批量测试所有病人的 LGE 序列
python batch_test_cine.py --model_alias qwen3-vl-235b --filter_sequence LGE

# 批量测试所有病人的 perfusion 序列
python batch_test_cine.py --model_alias qwen3-vl-235b --filter_sequence perfusion

# 批量测试所有病人的 T2 序列
python batch_test_cine.py --model_alias qwen3-vl-235b --filter_sequence T2

# 使用不同的模型
python batch_test_cine.py --model_alias gpt-4o --filter_sequence cine

# 从指定索引开始（用于断点续传）
python batch_test_cine.py --model_alias qwen3-vl-235b --filter_sequence cine --start_from 2
```

批量测试脚本会自动：
- 查找所有 `patient_*_vqa_png.json` 文件
- 对每个病人只测试指定序列类型的题目
- 将结果保存到 `output/{model_alias}/{patient_id}/` 目录
- 显示进度条和测试摘要

## 参数说明

### 必需参数
- `--json_path`: JSON数据文件路径

### 可选参数
- `--image_base_dir`: 图片基础目录（默认：当前目录）
- `--output_dir`: 输出目录（默认：`Heart_bench/output/{model_name}/`）
- `--output_format`: 输出格式，可选 `json`、`tsv` 或 `both`（默认：json）
- `--config_path`: 配置文件路径（默认：`Heart_bench/model_config.json`）
- `--include_reason`: 是否要求模型输出推理原因（默认：True）
- `--filter_sequence`: 过滤特定序列类型（如 `cine`、`LGE`、`perfusion`、`T2`），只测试匹配的题目

### 模型配置参数（二选一）

**方式1：使用模型别名（推荐）**
- `--test_model_alias`: 测试模型别名（从配置文件读取）
- `--judge_model_alias`: Judge模型别名（从配置文件读取）

**方式2：传统方式**
- `--test_model_type`: 测试模型类型，可选 `gpt`、`qwen`、`ksyun`（默认：gpt）
- `--test_model_name`: 测试模型名称（默认：gpt-4o）
- `--test_api_key`: 测试模型API密钥（也可通过环境变量设置）
- `--test_base_url`: 测试模型API基础URL（可选，用于代理）
- `--judge_model_type`: Judge模型类型（可选）
- `--judge_model_name`: Judge模型名称（默认：gpt-4o）
- `--judge_api_key`: Judge模型API密钥
- `--use_judge`: 是否使用Judge模型进行评估

## 支持的模型

### GPT模型（OpenAI）
- `gpt-4o`
- `gpt-4-vision-preview`
- `gpt-4-turbo`

### Qwen模型（阿里云DashScope）
- `qwen-vl-max`
- `qwen-vl-plus`

### Ksyun模型（金山云）
- `qwen3-vl-235b-a22b-thinking`（可通过别名 `qwen3-vl-235b` 使用）

## 支持的序列类型

框架支持以下序列类型，每个序列都有专门设计的prompt templates和reason analysis templates：

### Cine序列
- `cine_sax`: Cine序列短轴切面
  - 支持的字段：Thickening（增厚模式）
- `cine_4ch`: Cine序列四腔心切面
  - 支持的字段：Wall Motion Coordination（室壁运动协调性）、Wall Motion Amplitude（室壁运动幅度）、Systolic Function（收缩功能）、Diastolic Function（舒张功能）、Valves（瓣膜反流：二尖瓣、三尖瓣）、Effusion（积液：心包积液、胸腔积液）
- `cine_3ch`: Cine序列三腔心切面
  - 支持的字段：Valves（瓣膜反流：主动脉瓣）、Special Signs（特殊征象）

### LGE序列（Late Gadolinium Enhancement）
- `LGE_sax`: LGE序列短轴切面
  - 支持的字段：Enhancement Status（强化状态）、Abnormal Signal（异常信号）、High Signal Abnormal Region（高信号异常分区）、High Signal Distribution Pattern（高信号分布形状）、High Signal Myocardial Layer（高信号心肌层）、Low Signal Abnormal Region（低信号异常分区）、Low Signal Distribution Pattern（低信号分布形状）
- `LGE_4ch`: LGE序列四腔心切面
  - 支持的字段：High Signal Abnormal Segment（高信号异常节段）、Special Description（特殊描述）

### Perfusion序列（First-pass Myocardial Perfusion）
- `perfusion`: 灌注序列
  - 支持的字段：Perfusion Status（灌注状态）、Abnormal Segments（异常节段）、Abnormal Regions（异常区域）、Perfusion Abnormality Signal Characteristics（灌注异常信号特征：Reduced/Delayed/Defect）、Myocardial Layer（心肌层）

### T2序列（T2-weighted）
- `T2_sax`: T2序列短轴切面
  - 支持的字段：T2 Signal（T2信号）、Abnormal Segments（异常节段）、Abnormal Regions（异常区域）、Signal Distribution（信号分布）

每个序列的prompt template都包含：
- 明确的角色定义和任务说明
- 严格的输出格式要求（支持两行格式：Answer + Reason）
- 操作性的视觉定义（特别是多选项任务）
- 图像仅推理约束（不使用外部医学知识）

每个字段的reason template都提供：
- 5步结构化分析框架
- 具体的观察指导（告诉模型应该看什么）
- 图像特征描述要求（信号强度、位置、分布等）

## 输出文件

评估完成后，会在输出目录（默认：`Heart_bench/output/{model_name}/`）生成以下文件：

1. **detailed_results.json**: 每个问题的详细评估结果，包含：
   - `qid`: 问题ID
   - `field`: 字段名称
   - `patient_id`: 患者ID
   - `question_type`: 问题类型
   - `sequence_view`: 序列视图
   - `original_nii`: 原始NII文件路径
   - `gt`: 标准答案
   - `answer`: 模型答案
   - `reason`: 模型推理原因
   - `is_correct`: 是否正确
   - `accuracy`: 准确率
   - `hit`: 命中率（仅多选题）

2. **summary.json**: 总体统计摘要（按字段、序列视图、问题类型分组）

3. **report.txt**: 可读的文本报告

4. **detailed_results.tsv**: TSV格式的详细结果（如果指定了 `--output_format tsv` 或 `both`）

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
    model_alias='qwen3-vl-235b',
    config_manager=config_manager
)

# 加载数据（可以指定 filter_sequence 参数过滤特定序列）
data_loader = BenchmarkDataLoader(
    json_path='../dataset/patient_1322705_vqa_png.json',
    image_base_dir='../dataset',
    filter_sequence='cine'  # 只加载 cine 序列的题目
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
evaluator.save_results(result, output_format='both')  # 保存JSON和TSV格式

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

## Prompt Templates 和 Reason Templates

框架为每个序列类型和字段提供了专门设计的prompt templates和reason analysis templates，确保模型能够进行深入、有针对性的分析。

### Prompt Templates特性

1. **序列特定性**：每个序列（Cine、LGE、Perfusion、T2）都有针对其医学用途的专门prompt
2. **字段特定性**：每个字段都有针对其评估目标的专门prompt
3. **v2设计标准**：
   - 严格的两行输出格式（当`include_reason=True`时）
   - 多选题包含Z. None选项，避免在没有异常时强制选择
   - 图像仅推理约束，不使用外部医学知识或临床上下文
   - 操作性的视觉定义，特别是对于灌注序列的Reduced/Delayed/Defect分类

### Reason Templates特性

1. **5步分析框架**：每个reason template都包含5个关键分析步骤
2. **结构化指导**：明确告诉模型应该观察哪些特征、如何描述、如何比较
3. **图像特征聚焦**：专注于可见的图像特征（信号强度、位置、分布、连续性等）

### 使用示例

当使用`--include_reason`参数时，模型会收到包含reason template的完整prompt，例如：

```
You are a Vision-Language Model (VLM) for cardiac LGE MRI.
Task: Using ONLY the provided image frames, decide whether abnormal delayed enhancement is present...

When providing your reason, please follow this analysis framework:
1) Enhancement detection: whether any myocardium contains visually brighter foci...
2) Contrast comparison: compare suspected regions against adjacent normal myocardium...
3) Spatial consistency: confirm the bright area persists across adjacent frames...
4) Location cue: describe where it appears...
5) Conclusion: state present vs absent
```

详细的prompt和reason template设计文档请参考：[sequence_prompt_design.md](sequence_prompt_design.md)

## 注意事项

1. **图片路径**：确保JSON中的图片路径正确，且相对于`image_base_dir`
2. **API配额**：评估大量题目会消耗API配额，建议先用少量题目测试
3. **多图片输入**：每个问题可能包含多张图片（多个slice），模型需要支持多图输入
4. **Judge模型**：使用Judge模型会增加API调用成本，但评估更准确
5. **输出目录**：如果不指定`--output_dir`，结果会自动保存到`Heart_bench/output/{model_name}/`目录
6. **API密钥安全**：不要在配置文件中提交真实的API密钥，建议使用环境变量或`.gitignore`忽略配置文件
7. **序列过滤**：使用 `--filter_sequence` 参数时，过滤是基于 `sequence_view` 字段的字符串匹配（不区分大小写），例如 `cine` 会匹配 `cine_sax`、`cine_4ch`、`cine_3ch` 等
8. **批量测试**：批量测试脚本会自动查找所有 `patient_*_vqa_png.json` 文件，确保数据文件命名符合规范
9. **Prompt Templates**：框架会自动为支持的序列和字段选择对应的prompt template，如果字段不匹配，会使用通用prompt
10. **Reason输出**：使用`--include_reason`时，模型会被要求按照reason template的框架进行分析，输出更结构化的推理过程

## 更多信息

- 快速开始指南请参考 [QUICKSTART.md](QUICKSTART.md)
- 配置管理示例请参考 [example_config.py](example_config.py)
- 使用示例请参考 [example_usage.py](example_usage.py)
