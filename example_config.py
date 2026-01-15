"""
使用配置文件管理模型的示例
"""

from evaluate_benchmark import BenchmarkDataLoader, BenchmarkEvaluator
from model_apis import ModelFactory
from config_manager import ModelConfigManager

def example_with_config():
    """使用配置文件管理模型的示例"""
    print("=" * 80)
    print("示例: 使用配置文件管理模型")
    print("=" * 80)
    
    # 初始化配置管理器
    config_manager = ModelConfigManager()
    
    # 查看所有可用的模型
    print("\n可用的模型配置:")
    models = config_manager.list_models()
    for alias, config in models.items():
        print(f"  {alias}: {config.get('description', 'N/A')}")
        print(f"    类型: {config.get('type')}, 模型: {config.get('model')}")
    
    # 方式1: 使用模型别名创建模型（推荐）
    print("\n方式1: 使用模型别名 'qwen3-vl-235b'")
    model = ModelFactory.create_model(
        model_alias="qwen3-vl-235b",
        config_manager=config_manager
    )
    print(f"模型创建成功: {type(model).__name__}")
    
    # 方式2: 覆盖API key
    print("\n方式2: 使用模型别名并覆盖API key")
    model2 = ModelFactory.create_model(
        model_alias="qwen3-vl-235b",
        config_manager=config_manager,
        api_key="your-custom-api-key"  # 会覆盖配置文件中的API key
    )
    print(f"模型创建成功（使用自定义API key）")
    
    # 方式3: 添加新模型配置
    print("\n方式3: 添加新模型配置")
    config_manager.add_model("my-custom-model", {
        "type": "gpt",
        "model": "gpt-4o",
        "api_key": "your-api-key",
        "base_url": None,
        "max_tokens": 1000,
        "temperature": 0.1,
        "description": "我的自定义模型"
    })
    print("新模型配置已添加")
    
    # 使用新添加的模型
    custom_model = ModelFactory.create_model(
        model_alias="my-custom-model",
        config_manager=config_manager
    )
    print(f"自定义模型创建成功: {type(custom_model).__name__}")
    
    # 方式4: 在评估中使用
    print("\n方式4: 在评估中使用模型别名")
    data_loader = BenchmarkDataLoader(
        json_path='../1322705/patient_1322705_vqa_png.json',
        image_base_dir='../1322705'
    )
    
    evaluator = BenchmarkEvaluator(data_loader, model)
    
    # 只评估前2个问题作为示例
    questions = data_loader.load_questions()
    sample_questions = questions[:2]
    
    print(f"开始评估 {len(sample_questions)} 个问题...")
    result = evaluator.evaluate(sample_questions)
    
    print(f"\n评估完成！准确率: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")

if __name__ == "__main__":
    example_with_config()

