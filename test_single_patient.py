#!/usr/bin/env python3
"""
Test script for a single patient's all questions
Tests all questions for a single patient and checks for bugs
"""

import sys
import json
from pathlib import Path
from evaluate_benchmark import BenchmarkDataLoader, BenchmarkEvaluator, EvaluationResult
from model_apis import ModelFactory
from config_manager import ModelConfigManager
import argparse

def test_single_patient(patient_id: str, 
                       model_alias: str = None,
                       model_type: str = None,
                       model_name: str = None,
                       api_key: str = None,
                       max_workers: int = 1,
                       include_reason: bool = True,
                       output_dir: str = None):
    """
    Test all questions for a single patient
    
    Args:
        patient_id: Patient ID (e.g., "1322705")
        model_alias: Model alias from config
        model_type: Model type (gpt, qwen, ksyun)
        model_name: Model name
        api_key: API key
        max_workers: Number of concurrent workers
        include_reason: Whether to include reason
        output_dir: Output directory
    """
    # Find patient JSON file
    dataset_dir = Path(__file__).parent.parent / "dataset"
    
    # Try two possible locations:
    # 1. dataset/patient_{patient_id}_vqa_png.json (for most patients)
    # 2. dataset/{patient_id}/patient_{patient_id}_vqa_png.json (for patient 1322705)
    json_file = dataset_dir / f"patient_{patient_id}_vqa_png.json"
    if not json_file.exists():
        json_file = dataset_dir / patient_id / f"patient_{patient_id}_vqa_png.json"
    
    if not json_file.exists():
        print(f"Error: Patient JSON file not found for patient {patient_id}")
        print(f"Looking in: {dataset_dir}")
        print(f"Available files:")
        for f in dataset_dir.glob("patient_*_vqa_png.json"):
            print(f"  - {f.name}")
        for f in dataset_dir.rglob("patient_*_vqa_png.json"):
            if f.parent != dataset_dir:
                print(f"  - {f.relative_to(dataset_dir)}")
        return False
    
    print(f"Found patient JSON file: {json_file}")
    
    # Load and check questions
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total questions in file: {len(data)}")
    
    # Initialize config manager
    config_manager = ModelConfigManager()
    
    # Create test model
    test_model_kwargs = {}
    if api_key:
        test_model_kwargs['api_key'] = api_key
    
    if model_alias:
        test_model = ModelFactory.create_model(
            model_alias=model_alias,
            config_manager=config_manager,
            **test_model_kwargs
        )
    else:
        if not model_type:
            model_type = 'gpt'
        if not model_name:
            model_name = 'gpt-4o'
        
        test_model = ModelFactory.create_model(
            model_type=model_type,
            model=model_name,
            **test_model_kwargs
        )
    
    print(f"Using model: {test_model.model if hasattr(test_model, 'model') else 'Unknown'}")
    print(f"Concurrent workers: {max_workers}")
    
    # Initialize data loader
    # Determine image_base_dir based on JSON file location and image paths:
    # - If JSON is in dataset/patient_XXX.json, check image paths:
    #   - If paths are like "XXX/cine_sax/...", images are at dataset/XXX/XXX/... (nested)
    #   - If paths are like "cine_sax/...", images are at dataset/XXX/... (flat)
    # - If JSON is in dataset/XXX/patient_XXX.json, images are like "1322705/cine_sax_1/..." 
    #   -> use dataset/XXX (the loader will handle nested structure automatically)
    if json_file.parent == dataset_dir:
        # JSON file is directly in dataset/ directory
        # Check the first image path to determine structure
        if data and len(data) > 0 and 'image' in data[0]:
            first_img_path = data[0]['image'][0] if isinstance(data[0]['image'], list) else data[0]['image']
            # If path starts with patient_id, it's nested: dataset/1322705/1322705/...
            if first_img_path.startswith(f"{patient_id}/"):
                image_base_dir = dataset_dir / patient_id
                print(f"Detected nested structure: images at {image_base_dir}/...")
            else:
                # Flat structure: dataset/1322705/...
                image_base_dir = dataset_dir / patient_id
                print(f"Detected flat structure: images at {image_base_dir}/...")
        else:
            # Default: try nested structure
            image_base_dir = dataset_dir / patient_id
    else:
        # JSON file is in dataset/{patient_id}/ directory
        # The BenchmarkDataLoader will automatically try nested paths
        image_base_dir = json_file.parent
    
    print(f"Using image_base_dir: {image_base_dir}")
    data_loader = BenchmarkDataLoader(
        json_path=str(json_file),
        image_base_dir=str(image_base_dir),
        filter_sequence=None  # Test all sequences
    )
    
    # Load questions
    questions = data_loader.load_questions()
    print(f"Loaded {len(questions)} questions after filtering")
    
    if len(questions) == 0:
        print("Error: No questions loaded!")
        return False
    
    # Show question breakdown
    from collections import Counter
    sequence_counts = Counter(q.sequence_view for q in questions)
    field_counts = Counter(q.field for q in questions)
    type_counts = Counter(q.question_type for q in questions)
    
    print("\nQuestion breakdown:")
    print(f"  By sequence view: {dict(sequence_counts)}")
    print(f"  By field: {dict(field_counts)}")
    print(f"  By question type: {dict(type_counts)}")
    
    # Determine output directory
    if output_dir is None:
        model_name = model_alias or model_name or "unknown_model"
        base_output_dir = Path(__file__).parent / "output" / model_name
        output_dir = base_output_dir / patient_id
    else:
        output_dir = Path(output_dir)
    
    print(f"\nOutput directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create evaluator
    evaluator = BenchmarkEvaluator(
        data_loader=data_loader,
        test_model=test_model,
        include_reason=include_reason,
        output_dir=str(output_dir),
        max_workers=max_workers,
        resume=False  # Start fresh for testing
    )
    
    # Run evaluation
    print("\n" + "="*80)
    print("Starting evaluation...")
    print("="*80 + "\n")
    
    try:
        result = evaluator.evaluate(questions=questions)
        
        # Save results
        evaluator.save_results(result, str(output_dir), format='both')
        
        # Print summary
        print("\n" + "="*80)
        print("Evaluation Summary")
        print("="*80)
        print(f"Total questions: {result.total}")
        print(f"Correct answers: {result.correct}")
        print(f"Accuracy: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")
        print("\nBy Sequence View:")
        for seq, stats in sorted(result.per_sequence_view.items()):
            print(f"  {seq}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.4f})")
        print("\nBy Field:")
        for field, stats in sorted(result.per_field.items()):
            print(f"  {field}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.4f})")
        print("\nBy Question Type:")
        for qtype, stats in sorted(result.per_question_type.items()):
            print(f"  {qtype}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.4f})")
        print("="*80)
        
        # Check for potential bugs
        print("\n" + "="*80)
        print("Bug Check")
        print("="*80)
        
        bugs_found = []
        
        # Check 1: Empty answers
        empty_answers = [r for r in result.detailed_results if not r.get('answer', '').strip()]
        if empty_answers:
            bugs_found.append(f"Found {len(empty_answers)} questions with empty answers")
            print(f"Warning: {len(empty_answers)} questions have empty answers")
            for r in empty_answers[:5]:  # Show first 5
                print(f"    - QID: {r['qid']}, Field: {r['field']}")
        
        # Check 2: Missing reasons when include_reason=True
        if include_reason:
            missing_reasons = [r for r in result.detailed_results if not r.get('reason', '').strip()]
            if missing_reasons:
                bugs_found.append(f"Found {len(missing_reasons)} questions with missing reasons")
                print(f"Warning: {len(missing_reasons)} questions have missing reasons")
                for r in missing_reasons[:5]:  # Show first 5
                    print(f"    - QID: {r['qid']}, Field: {r['field']}")
        
        # Check 3: Inconsistent accuracy
        inconsistent = [r for r in result.detailed_results 
                       if (r.get('is_correct', False) and r.get('accuracy', 0.0) == 0.0) or
                          (not r.get('is_correct', False) and r.get('accuracy', 1.0) == 1.0)]
        if inconsistent:
            bugs_found.append(f"Found {len(inconsistent)} questions with inconsistent accuracy")
            print(f"Warning: {len(inconsistent)} questions have inconsistent accuracy/is_correct")
            for r in inconsistent[:5]:  # Show first 5
                print(f"    - QID: {r['qid']}, is_correct: {r['is_correct']}, accuracy: {r['accuracy']}")
        
        # Check 4: Missing hit for multiple choice
        missing_hit = [r for r in result.detailed_results 
                      if r.get('question_type') == 'Multiple Choice' and 'hit' not in r]
        if missing_hit:
            bugs_found.append(f"Found {len(missing_hit)} multiple choice questions without hit metric")
            print(f"Warning: {len(missing_hit)} multiple choice questions missing 'hit' metric")
        
        if not bugs_found:
            print("No obvious bugs found!")
        else:
            print(f"\nTotal issues found: {len(bugs_found)}")
        
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nError during evaluation: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Test all questions for a single patient')
    parser.add_argument('--patient_id', type=str, required=True,
                       help='Patient ID (e.g., "1322705")')
    parser.add_argument('--model_alias', type=str, default=None,
                       help='Model alias from config file')
    parser.add_argument('--model_type', type=str, default=None,
                       choices=['gpt', 'qwen', 'ksyun'],
                       help='Model type (if model_alias not provided)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name (if model_alias not provided)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key (optional, can override config)')
    parser.add_argument('--max_workers', type=int, default=1,
                       help='Number of concurrent workers (default: 1)')
    parser.add_argument('--include_reason', action='store_true', default=True,
                       help='Include reason in output (default: True)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: output/{model}/{patient_id})')
    
    args = parser.parse_args()
    
    success = test_single_patient(
        patient_id=args.patient_id,
        model_alias=args.model_alias,
        model_type=args.model_type,
        model_name=args.model_name,
        api_key=args.api_key,
        max_workers=args.max_workers,
        include_reason=args.include_reason,
        output_dir=args.output_dir
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
