"""
Medical Image VQA Benchmark Evaluation Framework
Supports different models such as GPT and Qwen
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import argparse
from tqdm import tqdm

try:
    from .prompt_manager import TestModelPromptGenerator, JudgeModelPromptGenerator
    from .model_apis import BaseModelAPI, ModelFactory
    from .config_manager import ModelConfigManager
    from .answer_parser import AnswerParser
except ImportError:
    from prompt_manager import TestModelPromptGenerator, JudgeModelPromptGenerator
    from model_apis import BaseModelAPI, ModelFactory
    from config_manager import ModelConfigManager
    from answer_parser import AnswerParser

# ==================== Data Class Definitions ====================
@dataclass
class Question:
    """Question data class"""
    qid: str
    field: str
    images: List[str]  # List of image paths
    question: str
    ground_truth: str
    is_multiple_choice: bool
    question_type: str
    sequence_view: str
    patient_id: str
    original_nii: Optional[str] = None

@dataclass
class EvaluationResult:
    """Evaluation result data class"""
    total: int
    correct: int
    accuracy: float
    per_field: Dict[str, Dict[str, float]]
    per_sequence_view: Dict[str, Dict[str, float]]
    per_question_type: Dict[str, Dict[str, float]]
    detailed_results: List[Dict[str, Any]]

# ==================== Data Loader ====================
class BenchmarkDataLoader:
    """Benchmark data loader"""
    
    def __init__(self, json_path: str, image_base_dir: str = "."):
        """
        Args:
            json_path: Path to JSON file
            image_base_dir: Base directory for images
        """
        self.json_path = json_path
        self.image_base_dir = Path(image_base_dir)
        
    def load_questions(self) -> List[Question]:
        """Load all questions"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        for item in data:
            # Get question and answer
            question_text = item['conversations'][0]['value']
            ground_truth = item['conversations'][1]['value']
            
            # Process image paths
            images = item.get('image', [])
            if isinstance(images, str):
                images = [images]
            
            # Build full image paths
            full_image_paths = []
            for img_path in images:
                full_path = self.image_base_dir / img_path
                if full_path.exists():
                    full_image_paths.append(str(full_path))
                else:
                    print(f"Warning: Image not found: {full_path}")
            
            if not full_image_paths:
                print(f"Warning: No valid images for qid {item['qid']}")
                continue
            
            question = Question(
                qid=item['qid'],
                field=item.get('field', ''),
                images=full_image_paths,
                question=question_text,
                ground_truth=ground_truth,
                is_multiple_choice=item.get('is_multiple_choice', False),
                question_type=item.get('question_type', ''),
                sequence_view=item.get('sequence_view', ''),
                patient_id=item.get('patient_id', ''),
                original_nii=item.get('original_nii')
            )
            questions.append(question)
        
        return questions

# ==================== 答案提取器 ====================
class AnswerExtractor:
    """从模型输出中提取答案"""
    
    @staticmethod
    def extract_answers(text: str, is_multiple_choice: bool = False) -> List[str]:
        """
        从文本中提取答案选项
        
        Args:
            text: 模型输出的文本
            is_multiple_choice: 是否为多选题
            
        Returns:
            提取的答案列表，如 ['A', 'B'] 或 ['A. Normal', 'B. Reduced']
        """
        # 提取所有选项字母（A-Z）
        pattern = r'\b([A-Z])\.?\s*[^\s]*(?:\s+[A-Z]\.?\s*[^\s]*)*'
        matches = re.findall(pattern, text)
        
        if matches:
            # Remove duplicates while preserving order
            seen = set()
            unique_matches = []
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    unique_matches.append(m)
            return unique_matches
        
        # If no letters found, try to extract full answers (e.g., "A. Normal")
        pattern_full = r'([A-Z])\.\s*([^\n;]+)'
        matches_full = re.findall(pattern_full, text)
        if matches_full:
            return [f"{m[0]}. {m[1].strip()}" for m in matches_full]
        
        return []
    
    @staticmethod
    def parse_ground_truth(gt_text: str) -> List[str]:
        """Parse ground truth answer"""
        gt_text = gt_text.strip()
        
        # First, try to match explicit format "A. Text" or "A, B. Text" or "A; B. Text"
        # Match patterns like "A. ", "A, B. ", "A; B. " at the start
        # This handles: "B. Thinned", "D. Paradoxical motion; B. Reduced"
        pattern_explicit = r'^([A-Z](?:[,;]\s*[A-Z])*)\.\s*'
        match = re.match(pattern_explicit, gt_text)
        if match:
            # Extract all letters from the match
            letters_str = match.group(1)
            letters = re.findall(r'([A-Z])', letters_str)
            if letters:
                return sorted(set(letters))
        
        # Try single letter at start: "A. Text"
        pattern_single = r'^([A-Z])\.\s*'
        match_single = re.match(pattern_single, gt_text)
        if match_single:
            return [match_single.group(1)]
        
        # Try to match comma/semicolon separated letters (e.g., "A, B" or "A; B")
        pattern_separated = r'^([A-Z](?:\s*[,;]\s*[A-Z])+)(?:\s|\.|$)'
        separated_match = re.match(pattern_separated, gt_text)
        if separated_match:
            letters_str = separated_match.group(1)
            letters = re.findall(r'([A-Z])', letters_str)
            if letters:
                return sorted(set(letters))
        
        # Fallback: extract capital letters that are followed by period, comma, semicolon, or end
        # This is more conservative to avoid matching letters in the middle of words
        pattern_fallback = r'\b([A-Z])(?:\s*[.,;]|\s*$)'
        letters = re.findall(pattern_fallback, gt_text)
        if letters:
            return sorted(set(letters))
        
        # If no letters found, return original text (remove option letter prefix)
        cleaned = re.sub(r'^[A-Z]\.\s*', '', gt_text)
        return [cleaned.strip()]

# ==================== Evaluator ====================
class AnswerEvaluator:
    """Answer evaluator"""
    
    @staticmethod
    def evaluate_single_choice(predicted: List[str], ground_truth: List[str]) -> Tuple[bool, Dict[str, float]]:
        """Evaluate single choice question"""
        if not predicted or not ground_truth:
            return False, {'accuracy': 0.0}
        
        # Compare option letters
        pred_letters = set([p[0] if len(p) > 0 else '' for p in predicted])
        gt_letters = set(ground_truth)
        
        is_correct = pred_letters == gt_letters
        return is_correct, {'accuracy': 1.0 if is_correct else 0.0}
    
    @staticmethod
    def evaluate_multiple_choice(predicted: List[str], ground_truth: List[str]) -> Tuple[bool, Dict[str, float]]:
        """
        Evaluate multiple choice question
        
        Returns:
            (is_correct, metrics) where metrics contains accuracy and hit
        """
        if not predicted or not ground_truth:
            return False, {'accuracy': 0.0, 'hit': 0.0}
        
        pred_set = set([p[0] if len(p) > 0 else '' for p in predicted])
        gt_set = set(ground_truth)
        
        if len(pred_set) == 0 and len(gt_set) == 0:
            return True, {'accuracy': 1.0, 'hit': 1.0}
        
        if len(pred_set) == 0 or len(gt_set) == 0:
            return False, {'accuracy': 0.0, 'hit': 0.0}
        
        # Exact match (accuracy)
        is_exact_match = pred_set == gt_set
        
        # Hit: whether at least one correct answer is selected
        intersection = pred_set & gt_set
        hit = 1.0 if len(intersection) > 0 else 0.0
        
        return is_exact_match, {'accuracy': 1.0 if is_exact_match else 0.0, 'hit': hit}
    
    @staticmethod
    def evaluate(predicted: List[str], ground_truth: List[str], is_multiple_choice: bool) -> Tuple[bool, Dict[str, float]]:
        """Unified evaluation interface"""
        if is_multiple_choice:
            return AnswerEvaluator.evaluate_multiple_choice(predicted, ground_truth)
        else:
            is_correct = AnswerEvaluator.evaluate_single_choice(predicted, ground_truth)
            return is_correct, {'accuracy': 1.0 if is_correct else 0.0}

# ==================== Main Evaluator Class ====================
class BenchmarkEvaluator:
    """Main benchmark evaluator class"""
    
    def __init__(self, 
                 data_loader: BenchmarkDataLoader,
                 test_model: BaseModelAPI,
                 judge_model: Optional[BaseModelAPI] = None,
                 use_judge: bool = False,
                 include_reason: bool = True):
        """
        Args:
            data_loader: Data loader
            test_model: Test model (for answering questions)
            judge_model: Judge model (for evaluating answers, optional)
            use_judge: Whether to use judge model for evaluation
            include_reason: Whether to require model to output reasoning
        """
        self.data_loader = data_loader
        self.test_model = test_model
        self.judge_model = judge_model
        self.use_judge = use_judge
        self.include_reason = include_reason
        self.answer_extractor = AnswerExtractor()
        self.evaluator = AnswerEvaluator()
        self.answer_parser = AnswerParser()
        
    def evaluate(self, questions: Optional[List[Question]] = None) -> EvaluationResult:
        """
        Execute evaluation
        
        Args:
            questions: List of questions, if None then load from data_loader
            
        Returns:
            Evaluation result
        """
        if questions is None:
            questions = self.data_loader.load_questions()
        
        # Statistics containers
        total = len(questions)
        correct = 0
        per_field = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        per_sequence_view = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        per_question_type = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        detailed_results = []
        
        # Iterate through all questions
        for question in tqdm(questions, desc="Evaluating"):
            # Generate test model prompt
            test_prompt = TestModelPromptGenerator.generate(
                sequence_view=question.sequence_view,
                question_type=question.question_type,
                is_multiple_choice=question.is_multiple_choice,
                question=question.question,
                field=question.field,
                include_reason=self.include_reason
            )
            
            # Model prediction
            try:
                raw_output = self.test_model.predict(question.images, test_prompt)
            except Exception as e:
                print(f"Error predicting for {question.qid}: {e}")
                raw_output = ""
                answer_text = ""
                reason = ""
            else:
                # Parse answer and reasoning
                answer_text, reason = self.answer_parser.parse_answer_with_reason(raw_output)
                # If parsing fails, use raw output
                if not answer_text:
                    answer_text = raw_output
            
            # Evaluate answer
            if self.use_judge and self.judge_model:
                # Use judge model for evaluation
                judge_prompt = JudgeModelPromptGenerator.generate(
                    sequence_view=question.sequence_view,
                    question_type=question.question_type,
                    is_multiple_choice=question.is_multiple_choice,
                    question=question.question,
                    ground_truth=question.ground_truth,
                    predicted_answer=answer_text
                )
                
                try:
                    judge_output = self.judge_model.predict([], judge_prompt)  # Judge doesn't need images
                    # Parse judge output (JSON format)
                    judgment = self._parse_judge_output(judge_output)
                    is_correct = judgment.get('is_correct', False)
                    # For judge model, calculate hit for multiple choice questions
                    if question.is_multiple_choice:
                        predicted_answers = self.answer_extractor.extract_answers(answer_text, question.is_multiple_choice)
                        gt_answers = self.answer_extractor.parse_ground_truth(question.ground_truth)
                        pred_set = set([p[0] if len(p) > 0 else '' for p in predicted_answers])
                        gt_set = set(gt_answers)
                        intersection = pred_set & gt_set
                        hit = 1.0 if len(intersection) > 0 else 0.0
                        metrics = {'accuracy': 1.0 if is_correct else 0.0, 'hit': hit}
                    else:
                        metrics = {'accuracy': 1.0 if is_correct else 0.0}
                except Exception as e:
                    print(f"Error in judge evaluation for {question.qid}: {e}")
                    # Fall back to rule-based evaluation
                    predicted_answers = self.answer_extractor.extract_answers(answer_text, question.is_multiple_choice)
                    gt_answers = self.answer_extractor.parse_ground_truth(question.ground_truth)
                    is_correct, metrics = self.evaluator.evaluate(predicted_answers, gt_answers, question.is_multiple_choice)
            else:
                # Use rule-based evaluation
                predicted_answers = self.answer_extractor.extract_answers(answer_text, question.is_multiple_choice)
                gt_answers = self.answer_extractor.parse_ground_truth(question.ground_truth)
                is_correct, metrics = self.evaluator.evaluate(predicted_answers, gt_answers, question.is_multiple_choice)
            
            if is_correct:
                correct += 1
            
            # Statistics
            per_field[question.field]['total'] += 1
            per_field[question.field]['correct'] += (1 if is_correct else 0)
            if question.is_multiple_choice:
                per_field[question.field]['hit'] += metrics.get('hit', 0.0)
            
            per_sequence_view[question.sequence_view]['total'] += 1
            per_sequence_view[question.sequence_view]['correct'] += (1 if is_correct else 0)
            if question.is_multiple_choice:
                per_sequence_view[question.sequence_view]['hit'] += metrics.get('hit', 0.0)
            
            per_question_type[question.question_type]['total'] += 1
            per_question_type[question.question_type]['correct'] += (1 if is_correct else 0)
            if question.is_multiple_choice:
                per_question_type[question.question_type]['hit'] += metrics.get('hit', 0.0)
            
            # Detailed results (according to user requirements)
            result_item = {
                'qid': question.qid,
                'field': question.field,
                'patient_id': question.patient_id,
                'question_type': question.question_type,
                'sequence_view': question.sequence_view,
                'original_nii': question.original_nii,
                'gt': question.ground_truth,  # ground truth in separate line
                'answer': answer_text,  # answer in separate line
                'reason': reason,  # reasoning
                'is_correct': is_correct,
                'accuracy': metrics.get('accuracy', 1.0 if is_correct else 0.0),
                'raw_output': raw_output  # Keep raw output for debugging
            }
            
            # Add hit for multiple choice questions
            if question.is_multiple_choice:
                result_item['hit'] = metrics.get('hit', 0.0)
            
            detailed_results.append(result_item)
        
        # Calculate average metrics
        for key in per_field:
            if per_field[key]['total'] > 0:
                per_field[key]['accuracy'] = per_field[key]['correct'] / per_field[key]['total']
                if 'hit' in per_field[key]:
                    per_field[key]['hit'] /= per_field[key]['total']
        
        for key in per_sequence_view:
            if per_sequence_view[key]['total'] > 0:
                per_sequence_view[key]['accuracy'] = per_sequence_view[key]['correct'] / per_sequence_view[key]['total']
                if 'hit' in per_sequence_view[key]:
                    per_sequence_view[key]['hit'] /= per_sequence_view[key]['total']
        
        for key in per_question_type:
            if per_question_type[key]['total'] > 0:
                per_question_type[key]['accuracy'] = per_question_type[key]['correct'] / per_question_type[key]['total']
                if 'hit' in per_question_type[key]:
                    per_question_type[key]['hit'] /= per_question_type[key]['total']
        
        return EvaluationResult(
            total=total,
            correct=correct,
            accuracy=correct / total if total > 0 else 0.0,
            per_field=dict(per_field),
            per_sequence_view=dict(per_sequence_view),
            per_question_type=dict(per_question_type),
            detailed_results=detailed_results
        )
    
    def _parse_judge_output(self, judge_output: str) -> Dict:
        """解析Judge模型的JSON输出"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{.*\}', judge_output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # If parsing fails, return default values
        return {'is_correct': False}
    
    def save_results(self, result: EvaluationResult, output_dir: str, format: str = "json"):
        """
        Save evaluation results
        
        Args:
            result: Evaluation result
            output_dir: Output directory
            format: Output format, "json" or "tsv" (tab-separated), default is JSON only
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results (JSON format)
        detailed_file = output_path / "detailed_results.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(result.detailed_results, f, ensure_ascii=False, indent=2)
        
        # Save TSV format (if specified)
        if format == "tsv" or format == "both":
            tsv_file = output_path / "detailed_results.tsv"
            self._save_tsv(result.detailed_results, tsv_file)
        
        # Save summary (JSON format)
        summary = {
            'overall': {
                'total': result.total,
                'correct': result.correct,
                'accuracy': result.accuracy
            },
            'per_field': result.per_field,
            'per_sequence_view': result.per_sequence_view,
            'per_question_type': result.per_question_type
        }
        
        summary_file = output_path / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Generate text report (optional)
        if format == "both":
            self._generate_text_report(result, output_path / "report.txt")
        
        print(f"\nEvaluation results saved to: {output_path}")
        print(f"  - Detailed results: {detailed_file}")
        print(f"  - Summary: {summary_file}")
        if format == "tsv" or format == "both":
            print(f"  - TSV format: {output_path / 'detailed_results.tsv'}")
        if format == "both":
            print(f"  - Text report: {output_path / 'report.txt'}")
    
    def _save_tsv(self, detailed_results: List[Dict], output_file: Path):
        """Save results in TSV format"""
        import csv
        
        # Define columns
        fieldnames = ['qid', 'field', 'patient_id', 'question_type', 'sequence_view', 
                     'original_nii', 'gt', 'answer', 'reason', 'is_correct', 'accuracy']
        
        # Add hit column if any multiple choice questions exist
        has_multiple_choice = any(item.get('question_type') == 'Multiple Choice' for item in detailed_results)
        if has_multiple_choice:
            fieldnames.append('hit')
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            
            for item in detailed_results:
                row = {
                    'qid': item.get('qid', ''),
                    'field': item.get('field', ''),
                    'patient_id': item.get('patient_id', ''),
                    'question_type': item.get('question_type', ''),
                    'sequence_view': item.get('sequence_view', ''),
                    'original_nii': item.get('original_nii', ''),
                    'gt': item.get('gt', ''),
                    'answer': item.get('answer', ''),
                    'reason': item.get('reason', ''),
                    'is_correct': item.get('is_correct', False),
                    'accuracy': item.get('accuracy', 0.0)
                }
                
                # Add hit for multiple choice questions
                if item.get('question_type') == 'Multiple Choice':
                    row['hit'] = item.get('hit', 0.0)
                elif has_multiple_choice:
                    row['hit'] = ''
                
                writer.writerow(row)
    
    def _generate_text_report(self, result: EvaluationResult, output_file: Path):
        """生成文本报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("医学影像VQA Benchmark评估报告\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            f.write("【Overall Statistics】\n")
            f.write(f"Total questions: {result.total}\n")
            f.write(f"Correct answers: {result.correct}\n")
            f.write(f"Accuracy: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)\n\n")
            
            # Statistics by field
            f.write("【Statistics by Field】\n")
            f.write("-" * 80 + "\n")
            for field, stats in sorted(result.per_field.items()):
                f.write(f"{field}:\n")
                f.write(f"  Total: {stats['total']}\n")
                f.write(f"  Correct: {stats['correct']}\n")
                f.write(f"  Accuracy: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)\n")
                if 'hit' in stats:
                    f.write(f"  Hit: {stats['hit']:.4f} ({stats['hit']*100:.2f}%)\n")
                f.write("\n")
            
            # Statistics by sequence view
            f.write("【Statistics by Sequence View】\n")
            f.write("-" * 80 + "\n")
            for view, stats in sorted(result.per_sequence_view.items()):
                f.write(f"{view}:\n")
                f.write(f"  Total: {stats['total']}\n")
                f.write(f"  Correct: {stats['correct']}\n")
                f.write(f"  Accuracy: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)\n")
                if 'hit' in stats:
                    f.write(f"  Hit: {stats['hit']:.4f} ({stats['hit']*100:.2f}%)\n")
                f.write("\n")
            
            # Statistics by question type
            f.write("【Statistics by Question Type】\n")
            f.write("-" * 80 + "\n")
            for qtype, stats in sorted(result.per_question_type.items()):
                f.write(f"{qtype}:\n")
                f.write(f"  Total: {stats['total']}\n")
                f.write(f"  Correct: {stats['correct']}\n")
                f.write(f"  Accuracy: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)\n")
                if 'hit' in stats:
                    f.write(f"  Hit: {stats['hit']:.4f} ({stats['hit']*100:.2f}%)\n")
                f.write("\n")

# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(description='Medical Image VQA Benchmark Evaluation')
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to JSON data file')
    parser.add_argument('--image_base_dir', type=str, default='.',
                       help='Base directory for images')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (if not specified, will use default path: Heart_bench/output/{model_name})')
    parser.add_argument('--test_model_alias', type=str, default=None,
                       help='Test model alias (read from config file, e.g., "qwen3-vl-235b")')
    parser.add_argument('--test_model_type', type=str, default=None,
                       choices=['gpt', 'qwen', 'ksyun', 'mog'],
                       help='Test model type (used if test_model_alias is not provided)')
    parser.add_argument('--test_model_name', type=str, default=None,
                       help='Test model name (used if test_model_alias is not provided)')
    parser.add_argument('--test_api_key', type=str, default=None,
                       help='Test model API key (will override config file settings)')
    parser.add_argument('--test_base_url', type=str, default=None,
                       help='Test model API base URL (will override config file settings)')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to config file (default: model_config.json)')
    parser.add_argument('--judge_model_alias', type=str, default=None,
                       help='Judge model alias (read from config file)')
    parser.add_argument('--judge_model_type', type=str, default=None,
                       choices=['gpt', 'qwen', 'ksyun', 'mog'],
                       help='Judge model type (used if judge_model_alias is not provided)')
    parser.add_argument('--judge_model_name', type=str, default=None,
                       help='Judge model name (used if judge_model_alias is not provided)')
    parser.add_argument('--judge_api_key', type=str, default=None,
                       help='Judge model API key (will override config file settings)')
    parser.add_argument('--use_judge', action='store_true',
                       help='Whether to use judge model for evaluation')
    parser.add_argument('--include_reason', action='store_true', default=True,
                       help='Whether to require model to output reasoning (default: True)')
    parser.add_argument('--output_format', type=str, default='json',
                       choices=['json', 'tsv', 'both'],
                       help='Output format: json, tsv, or both')
    
    args = parser.parse_args()
    
    # Initialize config manager
    config_manager = ModelConfigManager(config_path=args.config_path)
    
    # Create test model
    test_model_kwargs = {}
    if args.test_api_key:
        test_model_kwargs['api_key'] = args.test_api_key
    if args.test_base_url:
        test_model_kwargs['base_url'] = args.test_base_url
    
    if args.test_model_alias:
        # Use model alias
        test_model = ModelFactory.create_model(
            model_alias=args.test_model_alias,
            config_manager=config_manager,
            **test_model_kwargs
        )
    else:
        # Use traditional method
        if not args.test_model_type:
            # If not specified, use default model
            args.test_model_type = 'gpt'
        if not args.test_model_name:
            args.test_model_name = 'gpt-4o'
        
        test_model_config = {
            'model_type': args.test_model_type,
            'model': args.test_model_name,
            'api_key': args.test_api_key or os.getenv(f"{args.test_model_type.upper()}_API_KEY"),
        }
        if args.test_base_url:
            test_model_config['base_url'] = args.test_base_url
        
        test_model = ModelFactory.create_from_config(test_model_config)
    
    # Create judge model (if specified)
    judge_model = None
    if args.use_judge:
        judge_model_kwargs = {}
        if args.judge_api_key:
            judge_model_kwargs['api_key'] = args.judge_api_key
        
        if args.judge_model_alias:
            # Use model alias
            judge_model = ModelFactory.create_model(
                model_alias=args.judge_model_alias,
                config_manager=config_manager,
                **judge_model_kwargs
            )
        elif args.judge_model_type:
            # Use traditional method
            if not args.judge_model_name:
                args.judge_model_name = 'gpt-4o'
            
            judge_model_config = {
                'model_type': args.judge_model_type,
                'model': args.judge_model_name,
                'api_key': args.judge_api_key or os.getenv(f"{args.judge_model_type.upper()}_API_KEY"),
            }
            judge_model = ModelFactory.create_from_config(judge_model_config)
    
    # Initialize components
    data_loader = BenchmarkDataLoader(args.json_path, args.image_base_dir)
    evaluator = BenchmarkEvaluator(
        data_loader, 
        test_model, 
        judge_model=judge_model,
        use_judge=args.use_judge,
        include_reason=args.include_reason
    )
    
    # Determine output directory
    if args.output_dir is None:
        # Use default path: Heart_bench/output/{model_name}
        model_name = args.test_model_alias or args.test_model_name or "unknown_model"
        # Clean model name to ensure it can be used as folder name
        model_name = model_name.replace("/", "_").replace("\\", "_")
        base_output_dir = Path(__file__).parent / "output"
        output_dir = base_output_dir / model_name
    else:
        output_dir = Path(args.output_dir)
    
    print(f"Output directory: {output_dir}")
    
    # Execute evaluation
    print("Starting evaluation...")
    result = evaluator.evaluate()
    
    # Save results (default: JSON format only)
    evaluator.save_results(result, str(output_dir), format=args.output_format)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Total questions: {result.total}")
    print(f"Correct answers: {result.correct}")
    print(f"Accuracy: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()

