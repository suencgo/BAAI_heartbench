#!/usr/bin/env python3
"""
Script to fix incorrect is_correct and accuracy values in detailed_results.json files
"""

import json
import sys
from pathlib import Path

def fix_result_item(item):
    """Fix a single result item"""
    # Fix is_correct if it's in legacy format
    is_correct_raw = item.get('is_correct', False)
    if isinstance(is_correct_raw, list) and len(is_correct_raw) > 0:
        # Legacy format: [bool, {accuracy: ...}]
        is_correct = is_correct_raw[0] if isinstance(is_correct_raw[0], bool) else False
        item['is_correct'] = is_correct
        # Extract accuracy from the dict if present
        if len(is_correct_raw) > 1 and isinstance(is_correct_raw[1], dict):
            item['accuracy'] = is_correct_raw[1].get('accuracy', 0.0)
    elif isinstance(is_correct_raw, tuple):
        # Handle tuple format
        is_correct = is_correct_raw[0] if isinstance(is_correct_raw[0], bool) else False
        item['is_correct'] = is_correct
    else:
        is_correct = bool(is_correct_raw)
        item['is_correct'] = is_correct
    
    # Fix accuracy if it's inconsistent with is_correct
    accuracy = item.get('accuracy', 0.0)
    if is_correct and accuracy != 1.0:
        print(f"  Fixing QID {item.get('qid', 'unknown')}: is_correct=True but accuracy={accuracy}, setting to 1.0")
        item['accuracy'] = 1.0
    elif not is_correct and accuracy != 0.0:
        print(f"  Fixing QID {item.get('qid', 'unknown')}: is_correct=False but accuracy={accuracy}, setting to 0.0")
        item['accuracy'] = 0.0
    
    return item

def fix_results_file(json_file_path):
    """Fix a detailed_results.json file"""
    json_file = Path(json_file_path)
    if not json_file.exists():
        print(f"Error: File not found: {json_file_path}")
        return False
    
    print(f"Fixing: {json_file_path}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        fixed_count = 0
        for item in results:
            original_is_correct = item.get('is_correct')
            original_accuracy = item.get('accuracy')
            
            fixed_item = fix_result_item(item)
            
            if (original_is_correct != fixed_item.get('is_correct') or 
                original_accuracy != fixed_item.get('accuracy')):
                fixed_count += 1
        
        # Save fixed results
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"  Fixed {fixed_count} items")
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fix_results.py <detailed_results.json> [<detailed_results.json> ...]")
        print("Example: python3 fix_results.py ../output/qwen3-vl-235b/51622052/detailed_results.json")
        sys.exit(1)
    
    success_count = 0
    for json_file in sys.argv[1:]:
        if fix_results_file(json_file):
            success_count += 1
    
    print(f"\nFixed {success_count}/{len(sys.argv)-1} file(s)")

