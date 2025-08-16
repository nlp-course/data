#!/usr/bin/env python3
"""
Federalist Papers Complete Pipeline

This script orchestrates the complete Federalist Papers data processing and model training pipeline:

PHASE 1: Download raw data from Project Gutenberg
- Downloads the clean text version of the Federalist Papers
- Places files in Raw/ directory

PHASE 2: Prepare and process data
- Parses Gutenberg text into structured JSON
- Handles paper 70 duplication issue (Gutenberg has note, Yale had duplicate content)
- Builds HuggingFace tokenizer from training data only
- Splits data by author (Madison, Hamilton, disputed, others)
- Calculates keyword counts for lab 1-2 compatibility
- Saves processed data to Text/ directory

PHASE 3: Train language models
- Trains FFNN, RNN, UALM, ATTNLM, and Transformer models
- Uses consistent tokenization across all models
- Saves trained models to Models/ directory

Usage:
    # Run complete pipeline
    python Scripts/run_federalist_pipeline.py
    
    # Run specific phases
    python Scripts/run_federalist_pipeline.py --phase download
    python Scripts/run_federalist_pipeline.py --phase prepare
    python Scripts/run_federalist_pipeline.py --phase train

Prerequisites:
    - Python 3.8+ with required packages (see requirements.txt)
    - Internet connection for Gutenberg download
    - Sufficient disk space (~100MB for data, ~50MB for models)

Output Structure:
    Federalist/
    ├── Raw/           # Downloaded Gutenberg text files
    ├── Text/          # Processed JSON data and tokenizer
    └── Models/        # Trained language models
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    script_path = Path("Scripts") / script_name
    
    if not script_path.exists():
        print(f"❌ Error: Script not found: {script_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"PHASE: {description}")
    print(f"{'='*60}")
    print(f"Running: {script_path}")
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              check=True, 
                              capture_output=False)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ {description} failed with error: {e}")
        return False

def check_prerequisites():
    """Check if required directories and files exist."""
    print("Checking prerequisites...")
    
    # Check if we're in the right directory
    if not Path("Scripts").exists():
        print("❌ Error: Scripts directory not found. Run from Federalist/ directory.")
        return False
    
    # Check if required scripts exist
    required_scripts = [
        "download_federalist_from_gutenberg.py",
        "prepare_federalist_data.py", 
        "train_federalist_models.py"
    ]
    
    for script in required_scripts:
        if not (Path("Scripts") / script).exists():
            print(f"❌ Error: Required script not found: {script}")
            return False
    
    print("✅ All prerequisites satisfied")
    return True

def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Run the complete Federalist Papers pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python Scripts/run_federalist_pipeline.py
  
  # Run only data preparation
  python Scripts/run_federalist_pipeline.py --phase prepare
  
  # Run from data download through model training
  python Scripts/run_federalist_pipeline.py --phase download --phase prepare --phase train
        """
    )
    
    parser.add_argument(
        "--phase", 
        choices=["download", "prepare", "train"],
        action="append",
        help="Specific phase(s) to run (default: all phases)"
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip prerequisite checks"
    )
    
    args = parser.parse_args()
    
    print("Federalist Papers Complete Pipeline")
    print("="*50)
    
    # Check prerequisites unless skipped
    if not args.skip_checks and not check_prerequisites():
        sys.exit(1)
    
    # Determine which phases to run
    if args.phase:
        phases = args.phase
        print(f"Running specified phases: {', '.join(phases)}")
    else:
        phases = ["download", "prepare", "train"]
        print("Running all phases")
    
    # Phase definitions
    phase_configs = {
        "download": {
            "script": "download_federalist_from_gutenberg.py",
            "description": "Download raw data from Project Gutenberg"
        },
        "prepare": {
            "script": "prepare_federalist_data.py", 
            "description": "Prepare and process data"
        },
        "train": {
            "script": "train_federalist_models.py",
            "description": "Train language models"
        }
    }
    
    # Run phases
    success_count = 0
    for phase in phases:
        if phase in phase_configs:
            config = phase_configs[phase]
            if run_script(config["script"], config["description"]):
                success_count += 1
            else:
                print(f"\n❌ Pipeline failed at phase: {phase}")
                print("Check the error messages above and fix the issue before continuing.")
                sys.exit(1)
        else:
            print(f"⚠️  Warning: Unknown phase '{phase}' skipped")
    
    # Final summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Phases completed successfully: {success_count}/{len(phases)}")
    
    if success_count == len(phases):
        print("\n🎉 Complete pipeline finished successfully!")
        print("\nYour Federalist Papers data and models are ready for use in the labs.")
        print("\nNext steps:")
        print("1. Copy the Text/ and Models/ directories to your lab notebooks")
        print("2. Update lab paths to point to the new directory structure")
        print("3. Test the models and data in your labs")
    else:
        print(f"\n⚠️  Pipeline completed with {len(phases) - success_count} phase(s) failed")
        print("Check the error messages above and re-run failed phases.")

if __name__ == "__main__":
    main() 