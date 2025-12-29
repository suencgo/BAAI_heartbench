"""
Batch testing script for specific sequences across all patients
Supports filtering by sequence type: cine, LGE, perfusion, T2, etc.
"""
import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Configuration
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"
HEART_BENCH_DIR = SCRIPT_DIR

def get_patient_json_files():
    """Get all patient JSON file paths"""
    json_files = []
    
    # Find all patient_*_vqa_png.json files
    for json_file in DATASET_DIR.rglob("patient_*_vqa_png.json"):
        json_files.append(json_file)
    
    return sorted(json_files)

def test_patient_cine(json_file_path, model_alias="qwen3-vl-235b", filter_sequence="cine"):
    """Test specific sequences for a single patient"""
    # Determine image base directory
    # Images are referenced relative to dataset/ directory
    # For patient_1322705_vqa_png.json in dataset/1322705/, images are like "1322705/cine_sax_1/..."
    # For patient_51024050_vqa_png.json in dataset/, images are like "51024050/cine_sax/..."
    # So image_base_dir should always be DATASET_DIR
    image_base_dir = DATASET_DIR
    
    # Extract patient ID for output directory naming
    patient_id = json_file_path.stem.replace("patient_", "").replace("_vqa_png", "")
    
    # Build output directory: output/{model_alias}/{patient_id}/
    output_dir = OUTPUT_DIR / model_alias / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        str(HEART_BENCH_DIR / "evaluate_benchmark.py"),
        "--json_path", str(json_file_path),
        "--image_base_dir", str(image_base_dir),
        "--test_model_alias", model_alias,
        "--output_dir", str(output_dir),
        "--filter_sequence", filter_sequence,
        "--output_format", "json",
        "--include_reason"
    ]
    
    # Execute command
    try:
        result = subprocess.run(
            cmd, 
            cwd=str(HEART_BENCH_DIR), 
            check=True,
            capture_output=True,
            text=True
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def main():
    """Main function: batch test all patients"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch test specific sequences for all patients (cine, LGE, perfusion, T2, etc.)')
    parser.add_argument('--model_alias', type=str, default='qwen3-vl-235b',
                       help='Model alias to use (default: qwen3-vl-235b)')
    parser.add_argument('--filter_sequence', type=str, default='cine',
                       help='Sequence filter (default: cine)')
    parser.add_argument('--start_from', type=int, default=0,
                       help='Start from patient index (for resuming)')
    
    args = parser.parse_args()
    
    json_files = get_patient_json_files()
    
    if not json_files:
        print("No patient JSON files found!")
        return
    
    print(f"{'='*80}")
    print("Batch Testing Configuration")
    print(f"{'='*80}")
    print(f"Found {len(json_files)} patient JSON files")
    print(f"Model: {args.model_alias}")
    print(f"Filter sequence: {args.filter_sequence}")
    print(f"Output directory: {OUTPUT_DIR / args.model_alias}")
    print(f"Starting from patient index: {args.start_from}")
    print(f"{'='*80}\n")
    
    success_count = 0
    failed_patients = []
    
    # Process patients with progress bar
    for idx, json_file in enumerate(tqdm(json_files[args.start_from:], desc="Testing patients")):
        patient_id = json_file.stem.replace("patient_", "").replace("_vqa_png", "")
        actual_idx = idx + args.start_from
        
        print(f"\n[{actual_idx + 1}/{len(json_files)}] Testing patient: {patient_id}")
        print(f"JSON file: {json_file}")
        
        success, stdout, stderr = test_patient_cine(json_file, args.model_alias, args.filter_sequence)
        
        if success:
            success_count += 1
            print(f"✓ Successfully tested patient {patient_id}")
        else:
            failed_patients.append(patient_id)
            print(f"✗ Error testing patient {patient_id}")
            if stderr:
                print(f"Error output: {stderr[:500]}")  # Print first 500 chars of error
    
    # Print summary
    print(f"\n{'='*80}")
    print("Batch Testing Summary")
    print(f"{'='*80}")
    print(f"Total patients: {len(json_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_patients)}")
    if failed_patients:
        print(f"Failed patients: {', '.join(failed_patients)}")
    print(f"{'='*80}\n")
    
    # Print output locations
    print("Output files are saved in:")
    for patient_id in [f.stem.replace("patient_", "").replace("_vqa_png", "") for f in json_files]:
        output_path = OUTPUT_DIR / args.model_alias / patient_id
        if output_path.exists():
            print(f"  - {output_path}")

if __name__ == "__main__":
    main()

