#!/usr/bin/env python3
"""
Script to regenerate summary.json from detailed_results.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def regenerate_summary(detailed_results_file):
    """Regenerate summary.json from detailed_results.json"""
    detailed_file = Path(detailed_results_file)
    summary_file = detailed_file.parent / "summary.json"
    
    if not detailed_file.exists():
        print(f"Error: File not found: {detailed_results_file}")
        return False
    
    print(f"Reading: {detailed_file}")
    
    try:
        with open(detailed_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Statistics containers
        total = len(results)
        correct = sum(1 for r in results if r.get('is_correct', False))
        
        per_field = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        per_sequence_view = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        per_question_type = defaultdict(lambda: {'total': 0, 'correct': 0, 'hit': 0})
        
        # Process each result
        for result in results:
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
        
        # Calculate average metrics
        for key in per_field:
            if per_field[key]['total'] > 0:
                per_field[key]['accuracy'] = per_field[key]['correct'] / per_field[key]['total']
                if per_field[key]['total'] > 0 and 'hit' in per_field[key]:
                    per_field[key]['hit'] /= per_field[key]['total']
        
        for key in per_sequence_view:
            if per_sequence_view[key]['total'] > 0:
                per_sequence_view[key]['accuracy'] = per_sequence_view[key]['correct'] / per_sequence_view[key]['total']
                if per_sequence_view[key]['total'] > 0 and 'hit' in per_sequence_view[key]:
                    per_sequence_view[key]['hit'] /= per_sequence_view[key]['total']
        
        for key in per_question_type:
            if per_question_type[key]['total'] > 0:
                per_question_type[key]['accuracy'] = per_question_type[key]['correct'] / per_question_type[key]['total']
                if per_question_type[key]['total'] > 0 and 'hit' in per_question_type[key]:
                    per_question_type[key]['hit'] /= per_question_type[key]['total']
        
        # Create summary
        summary = {
            'overall': {
                'total': total,
                'correct': correct,
                'accuracy': correct / total if total > 0 else 0.0
            },
            'per_field': dict(per_field),
            'per_sequence_view': dict(per_sequence_view),
            'per_question_type': dict(per_question_type)
        }
        
        # Save summary
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Regenerated: {summary_file}")
        print(f"  Total: {total}, Correct: {correct}, Accuracy: {summary['overall']['accuracy']:.4f}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 regenerate_summary.py <detailed_results.json>")
        print("Example: python3 regenerate_summary.py ../output/qwen3-vl-235b/51622052/detailed_results.json")
        sys.exit(1)
    
    regenerate_summary(sys.argv[1])

