"""
使用示例：如何调用不同模型进行评估
"""

import os
try:
    from evaluate_benchmark import BenchmarkDataLoader, BenchmarkEvaluator
    from model_apis import ModelFactory
except ImportError:
    # 如果作为包导入
    from .evaluate_benchmark import BenchmarkDataLoader, BenchmarkEvaluator
    from .model_apis import ModelFactory

def example_gpt():
    """使用GPT模型评估的示例"""
    print("=" * 80)
    print("示例1: 使用GPT模型")
    print("=" * 80)
    
    # 创建GPT模型
    test_model = ModelFactory.create_model(
        model_type='gpt',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o',
        max_tokens=500
    )
    
    # 加载数据（注意路径相对于Heart_bench文件夹）
    data_loader = BenchmarkDataLoader(
        json_path='../1322705/patient_1322705_vqa_png.json',
        image_base_dir='../1322705'
    )
    
    # 创建评估器
    evaluator = BenchmarkEvaluator(data_loader, test_model)
    
    # 执行评估（可以只评估前几个问题作为示例）
    questions = data_loader.load_questions()
    sample_questions = questions[:5]  # 只评估前5个问题作为示例
    
    result = evaluator.evaluate(sample_questions)
    
    # 保存结果
    evaluator.save_results(result, './evaluation_results_gpt')
    
    print(f"\n评估完成！准确率: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")

def example_qwen():
    """使用Qwen模型评估的示例"""
    print("=" * 80)
    print("示例2: 使用Qwen模型")
    print("=" * 80)
    
    # 创建Qwen模型
    test_model = ModelFactory.create_model(
        model_type='qwen',
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model='qwen-vl-max',
        max_tokens=500
    )
    
    # 加载数据（注意路径相对于Heart_bench文件夹）
    data_loader = BenchmarkDataLoader(
        json_path='../1322705/patient_1322705_vqa_png.json',
        image_base_dir='../1322705'
    )
    
    # 创建评估器
    evaluator = BenchmarkEvaluator(data_loader, test_model)
    
    # 执行评估
    questions = data_loader.load_questions()
    sample_questions = questions[:5]  # 只评估前5个问题作为示例
    
    result = evaluator.evaluate(sample_questions)
    
    # 保存结果
    evaluator.save_results(result, './evaluation_results_qwen')
    
    print(f"\n评估完成！准确率: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")

def example_with_judge():
    """使用Judge模型评估的示例"""
    print("=" * 80)
    print("示例3: 使用Judge模型评估")
    print("=" * 80)
    
    # 创建测试模型
    test_model = ModelFactory.create_model(
        model_type='gpt',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o'
    )
    
    # 创建Judge模型
    judge_model = ModelFactory.create_model(
        model_type='gpt',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o'
    )
    
    # 加载数据（注意路径相对于Heart_bench文件夹）
    data_loader = BenchmarkDataLoader(
        json_path='../1322705/patient_1322705_vqa_png.json',
        image_base_dir='../1322705'
    )
    
    # 创建评估器（使用Judge模型）
    evaluator = BenchmarkEvaluator(
        data_loader, 
        test_model, 
        judge_model=judge_model,
        use_judge=True
    )
    
    # 执行评估
    questions = data_loader.load_questions()
    sample_questions = questions[:3]  # 只评估前3个问题作为示例（Judge会增加成本）
    
    result = evaluator.evaluate(sample_questions)
    
    # 保存结果
    evaluator.save_results(result, './evaluation_results_with_judge')
    
    print(f"\n评估完成！准确率: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")

def example_custom_config():
    """使用自定义配置的示例"""
    print("=" * 80)
    print("示例4: 使用自定义配置")
    print("=" * 80)
    
    # 从配置字典创建模型
    config = {
        'model_type': 'gpt',
        'model': 'gpt-4o',
        'api_key': os.getenv('OPENAI_API_KEY'),
        'base_url': None,  # 可以设置代理URL
        'max_tokens': 1000,
        'temperature': 0.1
    }
    
    test_model = ModelFactory.create_from_config(config)
    
    # 后续步骤相同...
    print("模型创建成功！")

if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not set")
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("Warning: DASHSCOPE_API_KEY not set")
    
    # 运行示例（取消注释以运行）
    # example_gpt()
    # example_qwen()
    # example_with_judge()
    # example_custom_config()
    
    print("\n请取消注释上面的示例函数来运行测试")

