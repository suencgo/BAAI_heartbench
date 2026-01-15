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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    
    def __init__(self, json_path: str, image_base_dir: str = ".", filter_sequence: str = None):
        """
        Args:
            json_path: Path to JSON file
            image_base_dir: Base directory for images
            filter_sequence: Filter questions by sequence_view (e.g., "cine" to match all cine sequences)
                            If None, load all questions
        """
        self.json_path = json_path
        self.image_base_dir = Path(image_base_dir)
        self.filter_sequence = filter_sequence
        
    def load_questions(self) -> List[Question]:
        """Load all questions"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        for item in data:
            # Filter by sequence_view if filter_sequence is specified
            sequence_view = item.get('sequence_view', '')
            if self.filter_sequence:
                if self.filter_sequence.lower() not in sequence_view.lower():
                    continue  # Skip questions that don't match the filter
            
            # Get question and answer
            question_text = item['conversations'][0]['value']
            ground_truth = item['conversations'][1]['value']
            
            # Process image paths
            images = item.get('image', [])
            if isinstance(images, str):
                images = [images]
            
            # Build full image paths
            # Try multiple path combinations to handle different directory structures
            full_image_paths = []
            for img_path in images:
                # Try 1: Direct path (image_base_dir / img_path)
                full_path = self.image_base_dir / img_path
                
                # Try 2: If path starts with patient_id, try nested structure
                # (e.g., if img_path is "1322705/cine_sax_1/..." and base is "dataset/1322705",
                #  try "dataset/1322705/1322705/cine_sax_1/...")
                if not full_path.exists() and "/" in str(img_path):
                    path_parts = str(img_path).split("/", 1)
                    if len(path_parts) == 2:
                        potential_patient_id = path_parts[0]
                        rest_path = path_parts[1]
                        # Try nested: base_dir / patient_id / rest_path
                        nested_path = self.image_base_dir / potential_patient_id / rest_path
                        if nested_path.exists():
                            full_path = nested_path
                
                # Try 3: If still not found, try removing patient_id prefix
                if not full_path.exists() and "/" in str(img_path):
                    path_parts = str(img_path).split("/", 1)
                    if len(path_parts) == 2 and path_parts[0].isdigit():
                        # Remove patient_id prefix and try flat structure
                        flat_path = self.image_base_dir / path_parts[1]
                        if flat_path.exists():
                            full_path = flat_path
                
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
            is_correct, metrics = AnswerEvaluator.evaluate_single_choice(predicted, ground_truth)
            return is_correct, metrics

# ==================== Main Evaluator Class ====================
class BenchmarkEvaluator:
    """Main benchmark evaluator class"""
    
    def __init__(self, 
                 data_loader: BenchmarkDataLoader,
                 test_model: BaseModelAPI,
                 judge_model: Optional[BaseModelAPI] = None,
                 use_judge: bool = False,
                 include_reason: bool = True,
                 output_dir: Optional[str] = None,
                 resume: bool = False,
                 max_workers: int = 1):
        """
        Args:
            data_loader: Data loader
            test_model: Test model (for answering questions)
            judge_model: Judge model (for evaluating answers, optional)
            use_judge: Whether to use judge model for evaluation
            include_reason: Whether to require model to output reasoning
            output_dir: Output directory for incremental saving
            resume: Whether to resume from existing results
            max_workers: Maximum number of concurrent workers (default: 1, sequential)
        """
        self.data_loader = data_loader
        self.test_model = test_model
        self.judge_model = judge_model
        self.use_judge = use_judge
        self.include_reason = include_reason
        self.answer_extractor = AnswerExtractor()
        self.evaluator = AnswerEvaluator()
        self.answer_parser = AnswerParser()
        self.output_dir = Path(output_dir) if output_dir else None
        self.resume = resume
        self.completed_qids = set()  # Track completed question IDs
        self.max_workers = max_workers
        self._save_lock = threading.Lock()  # Lock for thread-safe saving
        
    def _load_existing_results(self) -> Dict[str, Dict]:
        """Load existing results for resume functionality"""
        if not self.output_dir or not self.resume:
            return {}
        
        detailed_file = self.output_dir / "detailed_results.json"
        if not detailed_file.exists():
            return {}
        
        try:
            with open(detailed_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            # Create a dictionary indexed by qid
            results_dict = {item['qid']: item for item in existing_results}
            print(f"[Resume] Loaded {len(results_dict)} existing results from {detailed_file}")
            return results_dict
        except Exception as e:
            print(f"[Resume] Warning: Failed to load existing results: {e}")
            return {}
    
    def _save_single_result(self, result_item: Dict, all_results: List[Dict]):
        """Save a single result incrementally (thread-safe)"""
        if not self.output_dir:
            return
        
        with self._save_lock:
            # Check if this result already exists (for resume mode)
            existing_idx = None
            for i, existing_item in enumerate(all_results):
                if existing_item.get('qid') == result_item['qid']:
                    existing_idx = i
                    break
            
            if existing_idx is not None:
                # Update existing result
                all_results[existing_idx] = result_item
            else:
                # Add new result
                all_results.append(result_item)
            
            # Save detailed results
            detailed_file = self.output_dir / "detailed_results.json"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
    
    def _update_summary(self, all_results: List[Dict]):
        """Update summary statistics from all results"""
        if not self.output_dir:
            return
        
        # Recalculate statistics
        total = len(all_results)
        correct = sum(1 for r in all_results if r.get('is_correct', False))
        accuracy = correct / total if total > 0 else 0.0
        
        per_field = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        per_sequence_view = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        per_question_type = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        
        for result in all_results:
            field = result.get('field', '')
            seq_view = result.get('sequence_view', '')
            q_type = result.get('question_type', '')
            is_correct = result.get('is_correct', False)
            is_multiple = result.get('question_type') == 'Multiple Choice'
            
            per_field[field]['total'] += 1
            per_field[field]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_field[field]['hit'] += result.get('hit', 0.0)
            
            per_sequence_view[seq_view]['total'] += 1
            per_sequence_view[seq_view]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_sequence_view[seq_view]['hit'] += result.get('hit', 0.0)
            
            per_question_type[q_type]['total'] += 1
            per_question_type[q_type]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_question_type[q_type]['hit'] += result.get('hit', 0.0)
        
        # Calculate averages
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
        
        summary = {
            'overall': {
                'total': total,
                'correct': correct,
                'accuracy': accuracy
            },
            'per_field': dict(per_field),
            'per_sequence_view': dict(per_sequence_view),
            'per_question_type': dict(per_question_type)
        }
        
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
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
        
        # Load existing results if resuming
        existing_results = {}
        all_results = []
        if self.resume and self.output_dir:
            existing_results = self._load_existing_results()
            all_results = list(existing_results.values())
            self.completed_qids = set(existing_results.keys())
            print(f"[Resume] Found {len(self.completed_qids)} completed questions, will skip them")
        
        # Statistics containers
        total = len(questions)
        # Fix: Handle legacy format when counting correct
        def _get_is_correct(result):
            is_correct_raw = result.get('is_correct', False)
            if isinstance(is_correct_raw, list) and len(is_correct_raw) > 0:
                return bool(is_correct_raw[0])
            elif isinstance(is_correct_raw, tuple):
                return bool(is_correct_raw[0])
            else:
                return bool(is_correct_raw)
        correct = sum(1 for r in all_results if _get_is_correct(r))
        per_field = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        per_sequence_view = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        per_question_type = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        detailed_results = []  # Initialize detailed_results list
        
        # Initialize statistics from existing results
        for result in all_results:
            field = result.get('field', '')
            seq_view = result.get('sequence_view', '')
            q_type = result.get('question_type', '')
            
            # Fix: Handle legacy format where is_correct might be a list [bool, dict]
            is_correct_raw = result.get('is_correct', False)
            if isinstance(is_correct_raw, list) and len(is_correct_raw) > 0:
                # Legacy format: [bool, {accuracy: ...}]
                is_correct = is_correct_raw[0] if isinstance(is_correct_raw[0], bool) else False
                # Also fix the accuracy if it's wrong
                if len(is_correct_raw) > 1 and isinstance(is_correct_raw[1], dict):
                    result['accuracy'] = is_correct_raw[1].get('accuracy', 0.0)
                result['is_correct'] = is_correct  # Fix the format
            elif isinstance(is_correct_raw, tuple):
                # Handle tuple format (shouldn't happen in JSON, but just in case)
                is_correct = is_correct_raw[0] if isinstance(is_correct_raw[0], bool) else False
                result['is_correct'] = is_correct
            else:
                is_correct = bool(is_correct_raw)
            
            # Fix accuracy if it's inconsistent with is_correct
            accuracy = result.get('accuracy', 0.0)
            if is_correct and accuracy == 0.0:
                # If marked as correct but accuracy is 0, fix it
                result['accuracy'] = 1.0
            elif not is_correct and accuracy == 1.0:
                # If marked as incorrect but accuracy is 1, fix it
                result['accuracy'] = 0.0
            
            is_multiple = result.get('question_type') == 'Multiple Choice'
            
            per_field[field]['total'] += 1
            per_field[field]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_field[field]['hit'] += result.get('hit', 0.0)
            
            per_sequence_view[seq_view]['total'] += 1
            per_sequence_view[seq_view]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_sequence_view[seq_view]['hit'] += result.get('hit', 0.0)
            
            per_question_type[q_type]['total'] += 1
            per_question_type[q_type]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_question_type[q_type]['hit'] += result.get('hit', 0.0)
        
        # Process questions (sequential or concurrent)
        if self.max_workers > 1:
            # Concurrent processing
            print(f"[Concurrent] Using {self.max_workers} workers for parallel processing")
            detailed_results = self._evaluate_concurrent(questions, all_results, per_field, per_sequence_view, per_question_type)
        else:
            # Sequential processing (original behavior)
            detailed_results = self._evaluate_sequential(questions, all_results, per_field, per_sequence_view, per_question_type)
        
        # Update statistics from new detailed_results (only count new results, not existing ones)
        for result_item in detailed_results:
            # Skip if this was already counted in existing results
            if result_item['qid'] in existing_results:
                continue
                
            is_correct = result_item.get('is_correct', False)
            if is_correct and result_item.get('answer', '').strip():
                correct += 1
            
            field = result_item.get('field', '')
            seq_view = result_item.get('sequence_view', '')
            q_type = result_item.get('question_type', '')
            is_multiple = result_item.get('question_type') == 'Multiple Choice'
            
            per_field[field]['total'] += 1
            per_field[field]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_field[field]['hit'] += result_item.get('hit', 0.0)
            
            per_sequence_view[seq_view]['total'] += 1
            per_sequence_view[seq_view]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_sequence_view[seq_view]['hit'] += result_item.get('hit', 0.0)
            
            per_question_type[q_type]['total'] += 1
            per_question_type[q_type]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_question_type[q_type]['hit'] += result_item.get('hit', 0.0)
    
    def _evaluate_sequential(self, questions: List[Question], all_results: List[Dict],
                           per_field: Dict, per_sequence_view: Dict, per_question_type: Dict) -> List[Dict]:
        """Sequential evaluation (original behavior)"""
        detailed_results = []
        
        for question in tqdm(questions, desc="Evaluating"):
            # Skip if already completed and resuming
            if question.qid in self.completed_qids:
                print(f"[Resume] Skipping already completed QID: {question.qid}")
                continue
            # Generate test model prompt
            test_prompt = TestModelPromptGenerator.generate(
                sequence_view=question.sequence_view,
                question_type=question.question_type,
                is_multiple_choice=question.is_multiple_choice,
                question=question.question,
                field=question.field,
                include_reason=self.include_reason
            )
            
            # Debug: print prompt to check if reason is required
            if self.include_reason:
                print(f"[Prompt Check] include_reason=True, checking if prompt requires reason...")
                if "Reason:" in test_prompt or "reason:" in test_prompt.lower():
                    print(f"[Prompt Check] Prompt contains 'Reason:' requirement")
                else:
                    print(f"[Prompt Check] WARNING: Prompt does NOT contain 'Reason:' requirement!")
                    print(f"[Prompt Check] Prompt preview (last 200 chars): {test_prompt[-200:]}")
            
            # Model prediction
            print(f"\n{'='*80}")
            print(f"Processing QID: {question.qid}")
            print(f"Field: {question.field}, Sequence: {question.sequence_view}")
            print(f"Question Type: {question.question_type}, Multiple Choice: {question.is_multiple_choice}")
            print(f"Number of images: {len(question.images)}")
            print(f"Images: {[Path(img).name for img in question.images[:3]]}{'...' if len(question.images) > 3 else ''}")
            print(f"Prompt length: {len(test_prompt)} characters")
            print(f"{'='*80}")
            
            try:
                print(f"[API Call] Calling model.predict() for {question.qid}...")
                raw_output = self.test_model.predict(question.images, test_prompt)
                print(f"[API Response] Received response (length: {len(raw_output) if raw_output else 0} chars)")
                if raw_output:
                    print(f"[API Response] First 200 chars: {raw_output[:200]}...")
                else:
                    print(f"[API Response] WARNING: Empty response received!")
            except Exception as e:
                print(f"[ERROR] Exception during API call for {question.qid}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raw_output = ""
                answer_text = ""
                reason = ""
            else:
                # Parse answer and reasoning
                print(f"[Parsing] Parsing answer from response...")
                print(f"[Parsing] Raw output (first 500 chars): {raw_output[:500]}")
                print(f"[Parsing] Raw output length: {len(raw_output)}")
                answer_text, reason = self.answer_parser.parse_answer_with_reason(raw_output)
                print(f"[Parsing] Extracted answer: '{answer_text}', reason length: {len(reason)}")
                if not reason:
                    print(f"[Parsing] WARNING: No reason extracted from output!")
                    print(f"[Parsing] Raw output content: {repr(raw_output)}")
                    print(f"[Parsing] Checking if 'Reason:' or 'reason:' exists in output: {'Reason:' in raw_output or 'reason:' in raw_output}")
                # If parsing fails, use raw output
                if not answer_text:
                    print(f"[Parsing] WARNING: Failed to extract answer, using raw output")
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
                
                # Debug: print extraction results
                print(f"[Evaluation] QID: {question.qid}")
                print(f"[Evaluation] Answer text: '{answer_text}'")
                print(f"[Evaluation] Extracted predicted answers: {predicted_answers}")
                print(f"[Evaluation] Extracted ground truth: {gt_answers}")
                
                is_correct, metrics = self.evaluator.evaluate(predicted_answers, gt_answers, question.is_multiple_choice)
                
                print(f"[Evaluation] Is correct: {is_correct}, Metrics: {metrics}")
            
            # Only count as correct if answer_text is not empty
            if is_correct and (not answer_text or not answer_text.strip()):
                print(f"[Evaluation WARNING] QID {question.qid} marked as correct but answer is empty! Setting to incorrect.")
                is_correct = False
                metrics['accuracy'] = 0.0
                if question.is_multiple_choice:
                    metrics['hit'] = 0.0
            
            # Detailed results (according to user requirements)
            # Ensure accuracy is consistent with is_correct
            accuracy = metrics.get('accuracy', 1.0 if is_correct else 0.0)
            if is_correct and accuracy != 1.0:
                accuracy = 1.0
            elif not is_correct and accuracy != 0.0:
                accuracy = 0.0
            
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
                'accuracy': accuracy,  # Ensure consistency with is_correct
                'raw_output': raw_output  # Keep raw output for debugging
            }
            
            # Add hit for multiple choice questions
            if question.is_multiple_choice:
                result_item['hit'] = metrics.get('hit', 0.0)
            
            # Add to both detailed_results (for return) and all_results (for incremental save)
            detailed_results.append(result_item)
            
            # Incremental save: save each result immediately
            if self.output_dir:
                self._save_single_result(result_item, all_results)
                self._update_summary(all_results)
                self.completed_qids.add(question.qid)
        
        return detailed_results
    
    def _evaluate_concurrent(self, questions: List[Question], all_results: List[Dict],
                            per_field: Dict, per_sequence_view: Dict, per_question_type: Dict) -> List[Dict]:
        """Concurrent evaluation using ThreadPoolExecutor"""
        detailed_results = []
        completed_lock = threading.Lock()
        
        def process_question(question: Question) -> Optional[Dict]:
            """Process a single question (worker function)"""
            # Skip if already completed and resuming
            with completed_lock:
                if question.qid in self.completed_qids:
                    return None
            
            try:
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
                    print(f"[{question.qid}] ERROR: {type(e).__name__}: {e}")
                    raw_output = ""
                    answer_text = ""
                    reason = ""
                else:
                    # Parse answer and reasoning
                    answer_text, reason = self.answer_parser.parse_answer_with_reason(raw_output)
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
                        judge_output = self.judge_model.predict([], judge_prompt)
                        judgment = self._parse_judge_output(judge_output)
                        is_correct = judgment.get('is_correct', False)
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
                        print(f"[{question.qid}] Judge error: {e}, falling back to rule-based")
                        predicted_answers = self.answer_extractor.extract_answers(answer_text, question.is_multiple_choice)
                        gt_answers = self.answer_extractor.parse_ground_truth(question.ground_truth)
                        is_correct, metrics = self.evaluator.evaluate(predicted_answers, gt_answers, question.is_multiple_choice)
                else:
                    # Use rule-based evaluation
                    predicted_answers = self.answer_extractor.extract_answers(answer_text, question.is_multiple_choice)
                    gt_answers = self.answer_extractor.parse_ground_truth(question.ground_truth)
                    is_correct, metrics = self.evaluator.evaluate(predicted_answers, gt_answers, question.is_multiple_choice)
                
                # Only count as correct if answer_text is not empty
                if is_correct and (not answer_text or not answer_text.strip()):
                    is_correct = False
                    metrics['accuracy'] = 0.0
                    if question.is_multiple_choice:
                        metrics['hit'] = 0.0
                
                # Ensure accuracy is consistent with is_correct
                accuracy = metrics.get('accuracy', 1.0 if is_correct else 0.0)
                if is_correct and accuracy != 1.0:
                    accuracy = 1.0
                elif not is_correct and accuracy != 0.0:
                    accuracy = 0.0
                
                result_item = {
                    'qid': question.qid,
                    'field': question.field,
                    'patient_id': question.patient_id,
                    'question_type': question.question_type,
                    'sequence_view': question.sequence_view,
                    'original_nii': question.original_nii,
                    'gt': question.ground_truth,
                    'answer': answer_text,
                    'reason': reason,
                    'is_correct': is_correct,
                    'accuracy': accuracy,
                    'raw_output': raw_output
                }
                
                if question.is_multiple_choice:
                    result_item['hit'] = metrics.get('hit', 0.0)
                
                # Incremental save: save each result immediately
                if self.output_dir:
                    self._save_single_result(result_item, all_results)
                    self._update_summary(all_results)
                    with completed_lock:
                        self.completed_qids.add(question.qid)
                
                return result_item
                
            except Exception as e:
                print(f"[{question.qid}] Exception: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # Filter out already completed questions
        questions_to_process = [q for q in questions if q.qid not in self.completed_qids]
        
        # Process questions concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_question = {executor.submit(process_question, q): q for q in questions_to_process}
            
            for future in tqdm(as_completed(future_to_question), total=len(questions_to_process), desc="Evaluating"):
                question = future_to_question[future]
                try:
                    result_item = future.result()
                    if result_item:
                        detailed_results.append(result_item)
                except Exception as e:
                    print(f"[{question.qid}] Future exception: {e}")
        
        # Merge all_results with detailed_results (in case of resume)
        if self.resume and self.output_dir:
            new_results_map = {item['qid']: item for item in detailed_results}
            for i, existing_item in enumerate(all_results):
                if existing_item['qid'] in new_results_map:
                    all_results[i] = new_results_map[existing_item['qid']]
            for new_item in detailed_results:
                if new_item['qid'] not in {item['qid'] for item in all_results}:
                    all_results.append(new_item)
            detailed_results = all_results
        
        return detailed_results
    
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
        
        # Load existing results if resuming
        existing_results = {}
        all_results = []
        if self.resume and self.output_dir:
            existing_results = self._load_existing_results()
            all_results = list(existing_results.values())
            self.completed_qids = set(existing_results.keys())
            print(f"[Resume] Found {len(self.completed_qids)} completed questions, will skip them")
        
        # Statistics containers
        total = len(questions)
        def _get_is_correct(result):
            is_correct_raw = result.get('is_correct', False)
            if isinstance(is_correct_raw, list) and len(is_correct_raw) > 0:
                return bool(is_correct_raw[0])
            elif isinstance(is_correct_raw, tuple):
                return bool(is_correct_raw[0])
            else:
                return bool(is_correct_raw)
        correct = sum(1 for r in all_results if _get_is_correct(r))
        per_field = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        per_sequence_view = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        per_question_type = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        
        # Initialize statistics from existing results
        for result in all_results:
            field = result.get('field', '')
            seq_view = result.get('sequence_view', '')
            q_type = result.get('question_type', '')
            
            is_correct_raw = result.get('is_correct', False)
            if isinstance(is_correct_raw, list) and len(is_correct_raw) > 0:
                is_correct = is_correct_raw[0] if isinstance(is_correct_raw[0], bool) else False
                if len(is_correct_raw) > 1 and isinstance(is_correct_raw[1], dict):
                    result['accuracy'] = is_correct_raw[1].get('accuracy', 0.0)
                result['is_correct'] = is_correct
            elif isinstance(is_correct_raw, tuple):
                is_correct = is_correct_raw[0] if isinstance(is_correct_raw[0], bool) else False
                result['is_correct'] = is_correct
            else:
                is_correct = bool(is_correct_raw)
            
            accuracy = result.get('accuracy', 0.0)
            if is_correct and accuracy == 0.0:
                result['accuracy'] = 1.0
            elif not is_correct and accuracy == 1.0:
                result['accuracy'] = 0.0
            
            is_multiple = result.get('question_type') == 'Multiple Choice'
            
            per_field[field]['total'] += 1
            per_field[field]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_field[field]['hit'] += result.get('hit', 0.0)
            
            per_sequence_view[seq_view]['total'] += 1
            per_sequence_view[seq_view]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_sequence_view[seq_view]['hit'] += result.get('hit', 0.0)
            
            per_question_type[q_type]['total'] += 1
            per_question_type[q_type]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_question_type[q_type]['hit'] += result.get('hit', 0.0)
        
        # Process questions (sequential or concurrent)
        if self.max_workers > 1:
            # Concurrent processing
            print(f"[Concurrent] Using {self.max_workers} workers for parallel processing")
            detailed_results = self._evaluate_concurrent(questions, all_results, per_field, per_sequence_view, per_question_type)
        else:
            # Sequential processing (original behavior)
            detailed_results = self._evaluate_sequential(questions, all_results, per_field, per_sequence_view, per_question_type)
        
        # Update statistics from detailed_results
        for result_item in detailed_results:
            is_correct = result_item.get('is_correct', False)
            if is_correct and result_item.get('answer', '').strip():
                correct += 1
            
            field = result_item.get('field', '')
            seq_view = result_item.get('sequence_view', '')
            q_type = result_item.get('question_type', '')
            is_multiple = result_item.get('question_type') == 'Multiple Choice'
            
            per_field[field]['total'] += 1
            per_field[field]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_field[field]['hit'] += result_item.get('hit', 0.0)
            
            per_sequence_view[seq_view]['total'] += 1
            per_sequence_view[seq_view]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_sequence_view[seq_view]['hit'] += result_item.get('hit', 0.0)
            
            per_question_type[q_type]['total'] += 1
            per_question_type[q_type]['correct'] += (1 if is_correct else 0)
            if is_multiple:
                per_question_type[q_type]['hit'] += result_item.get('hit', 0.0)
        
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
    parser.add_argument('--filter_sequence', type=str, default=None,
                       help='Filter questions by sequence_view (e.g., "cine" to test only cine sequences, "LGE" for LGE sequences)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume evaluation from existing results (skip already completed questions)')
    parser.add_argument('--max_workers', type=int, default=1,
                       help='Maximum number of concurrent workers for parallel processing (default: 1, sequential)')
    
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
    data_loader = BenchmarkDataLoader(
        args.json_path, 
        args.image_base_dir,
        filter_sequence=args.filter_sequence
    )
    evaluator = BenchmarkEvaluator(
        data_loader, 
        test_model, 
        judge_model=judge_model,
        use_judge=args.use_judge,
        include_reason=args.include_reason,
        max_workers=args.max_workers
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
    if args.resume:
        print("Resume mode: Will skip already completed questions")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update evaluator with output_dir and resume flag
    evaluator.output_dir = Path(output_dir)
    evaluator.resume = args.resume
    
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

