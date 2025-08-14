#!/usr/bin/env python3
"""
Federalist Papers Data and Model Builder

This script replaces the auxiliary Jupyter notebook and provides a complete pipeline to:
1. Download raw Federalist Papers data
2. Build a HuggingFace tokenizer
3. Tokenize the data
4. Train all language models (FFNN, RNN, UALM, ATTNLM, Transformer)
5. Save everything to data/ directory for use in labs

Usage:
    python build_federalist_data.py
"""

import os
import json
import math
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
import argparse

import torch

# Path constants - when run from Data repo root
DEFAULT_OUTPUT_DIR = "."  # Output to current directory (Data repo root)
FEDERALIST_BASE_DIR = "Federalist"
RAW_SUBDIR = "Raw"
TEXT_SUBDIR = "Text"
MODELS_SUBDIR = "Models"

# Full paths (will be combined with output_dir)
RAW_DIR_PATH = os.path.join(FEDERALIST_BASE_DIR, RAW_SUBDIR)
TEXT_DIR_PATH = os.path.join(FEDERALIST_BASE_DIR, TEXT_SUBDIR)
MODELS_DIR_PATH = os.path.join(FEDERALIST_BASE_DIR, MODELS_SUBDIR)

# File names
FEDERALIST_PAPERS_JSON = "federalist_papers.json"
PROCESSED_DATA_JSON = "federalist_data_processed.json"
TOKENIZER_FILE = "tokenizer.pt"
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

# Import model classes
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuration
SEED = 1234
MIN_FREQ = 3
TRAIN_RATIO = 0.9

# Model hyperparameters
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
N_GRAM = 5
EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
SEQUENCE_LENGTH = 35

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)

def download_raw_data(output_dir=DEFAULT_OUTPUT_DIR):
    """Download raw Federalist Papers data from the nlp-course data repo."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for the organized structure
    text_data_file = os.path.join(output_dir, TEXT_DIR_PATH, FEDERALIST_PAPERS_JSON)
    
    if os.path.exists(text_data_file):
        print(f"Clean text data already exists: {text_data_file}")
        return
    
    print("Clean text data not found, downloading and organizing from nlp-course data repo...")
    
    # Import and run the download script
    import subprocess
    import sys
    
    try:
        # Run the download script
        result = subprocess.run([
            sys.executable, 
            os.path.join(os.path.dirname(__file__), "download_federalist_from_repo.py")
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("✓ Download and organization completed successfully")
            if result.stdout:
                print("Download output:")
                print(result.stdout)
        else:
            print(f"✗ Download failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            raise Exception("Download script failed")
            
    except Exception as e:
        print(f"Failed to run download script: {e}")
        print("Creating placeholder data as fallback...")
        # Create fallback in the new structure
        os.makedirs(os.path.join(output_dir, TEXT_DIR_PATH), exist_ok=True)
        create_placeholder_data(text_data_file)

def create_placeholder_data(filepath):
    """Create placeholder data structure until we get the real raw data."""
    # This is a temporary placeholder - replace with actual raw text processing
    placeholder_data = [
        {
            "number": i+1,
            "title": f"Federalist Paper {i+1}",
            "authors": "Hamilton" if i % 2 == 0 else "Madison",
            "text": f"This is a placeholder for Federalist Paper {i+1}. " * 100  # Placeholder text
        }
        for i in range(85)  # 85 Federalist Papers
    ]
    
    with open(filepath, 'w') as f:
        json.dump(placeholder_data, f, indent=2)

def load_raw_data(data_dir=DEFAULT_OUTPUT_DIR):
    """Load raw Federalist Papers data from the organized structure."""
    filepath = os.path.join(data_dir, TEXT_DIR_PATH, FEDERALIST_PAPERS_JSON)
    
    if not os.path.exists(filepath):
        print(f"Clean text data file not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} Federalist Papers from organized structure")
    return data

def tokenize_text(text):
    """Simple tokenization - split on whitespace and basic punctuation."""
    import re
    # Convert to lowercase and split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    return tokens

def build_tokenizer(raw_data):
    """Build HuggingFace tokenizer from raw text data."""
    print("Building tokenizer from raw text...")
    
    # Collect all text and tokenize
    all_tokens = []
    for doc in raw_data:
        if 'text' in doc:
            tokens = tokenize_text(doc['text'])
            all_tokens.extend(tokens)
    
    print(f"Total tokens before filtering: {len(all_tokens)}")
    
    # Count token frequencies and filter by MIN_FREQ
    token_counts = Counter(all_tokens)
    vocab_tokens = [token for token, count in token_counts.items() if count >= MIN_FREQ]
    
    print(f"Vocabulary size (min_freq={MIN_FREQ}): {len(vocab_tokens)}")
    
    # Create HuggingFace tokenizer
    word_tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    word_tokenizer.pre_tokenizer = WhitespaceSplit()
    
    # Add tokens with single_word=True to prevent unexpected tokenization
    token_list = [AddedToken(token, single_word=True) for token in vocab_tokens]
    word_tokenizer.add_tokens(token_list)
    
    # Create PreTrainedTokenizerFast (same as in labs)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=word_tokenizer,
        pad_token='[PAD]',
        unk_token='[UNK]'
    )
    
    print(f"Final tokenizer vocabulary size: {len(hf_tokenizer)}")
    return hf_tokenizer

def process_and_split_data(raw_data, tokenizer):
    """Process raw data and split into train/validation/test sets."""
    print("Processing and splitting data...")
    
    # Tokenize all documents
    tokenized_docs = []
    for doc in raw_data:
        if 'text' in doc:
            tokens = tokenize_text(doc['text'])
            tokenized_doc = {
                'number': doc['number'],
                'title': doc['title'],
                'authors': doc['authors'],
                'tokens': tokens
            }
            tokenized_docs.append(tokenized_doc)
    
    # Split by author and create train/validation/test sets
    madison_docs = [doc for doc in tokenized_docs if doc['authors'] == 'Madison']
    hamilton_docs = [doc for doc in tokenized_docs if doc['authors'] == 'Hamilton']
    unknown_docs = [doc for doc in tokenized_docs if 'Hamilton or Madison' in doc['authors']]
    
    # For test set, use unknown authorship papers (assign to Madison as in original)
    test_docs = unknown_docs
    for doc in test_docs:
        doc['authors'] = 'Madison'
    
    # Split Madison papers
    random.shuffle(madison_docs)
    madison_train_size = int(math.floor(TRAIN_RATIO * len(madison_docs)))
    madison_train = madison_docs[:madison_train_size]
    madison_val = madison_docs[madison_train_size:]
    
    # Split Hamilton papers
    random.shuffle(hamilton_docs)
    hamilton_train_size = int(math.floor(TRAIN_RATIO * len(hamilton_docs)))
    hamilton_train = hamilton_docs[:hamilton_train_size]
    hamilton_val = hamilton_docs[hamilton_train_size:]
    
    # Extract tokens
    train_tokens_m = [token for doc in madison_train for token in doc['tokens']]
    valid_tokens_m = [token for doc in madison_val for token in doc['tokens']]
    train_tokens_h = [token for doc in hamilton_train for token in doc['tokens']]
    valid_tokens_h = [token for doc in hamilton_val for token in doc['tokens']]
    test_tokens = [token for doc in test_docs for token in doc['tokens']]
    
    # Make train/valid sizes match for Hamilton and Madison
    min_train_size = min(len(train_tokens_m), len(train_tokens_h))
    min_val_size = min(len(valid_tokens_m), len(valid_tokens_h))
    
    train_tokens_m = train_tokens_m[:min_train_size]
    train_tokens_h = train_tokens_h[:min_train_size]
    valid_tokens_m = valid_tokens_m[:min_val_size]
    valid_tokens_h = valid_tokens_h[:min_val_size]
    
    print(f"Madison - Training: {len(train_tokens_m)} tokens, Validation: {len(valid_tokens_m)} tokens")
    print(f"Hamilton - Training: {len(train_tokens_h)} tokens, Validation: {len(valid_tokens_h)} tokens")
    print(f"Test: {len(test_tokens)} tokens")
    
    return {
        'train_tokens_m': train_tokens_m,
        'train_tokens_h': train_tokens_h,
        'valid_tokens_m': valid_tokens_m,
        'valid_tokens_h': valid_tokens_h,
        'test_tokens': test_tokens,
        'tokenized_docs': tokenized_docs
    }

def save_processed_data(data, tokenizer, output_dir=DEFAULT_OUTPUT_DIR):
    """Save processed data and tokenizer to the organized structure."""
    # Create necessary directories
    models_dir = os.path.join(output_dir, MODELS_DIR_PATH)
    text_dir = os.path.join(output_dir, TEXT_DIR_PATH)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    
    print("Saving processed data and tokenizer...")
    
    # Save tokenizer to Models directory
    tokenizer_path = os.path.join(models_dir, TOKENIZER_FILE)
    torch.save(tokenizer, tokenizer_path)
    print(f"  ✓ Saved tokenizer to {tokenizer_path}")
    
    # Save processed data as JSON to Text directory
    processed_data = {
        'documents': data['tokenized_docs'],
        'splits': {
            'train_tokens_m': data['train_tokens_m'],
            'train_tokens_h': data['train_tokens_h'],
            'valid_tokens_m': data['valid_tokens_m'],
            'valid_tokens_h': data['valid_tokens_h'],
            'test_tokens': data['test_tokens']
        }
    }
    
    data_path = os.path.join(text_dir, PROCESSED_DATA_JSON)
    with open(data_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    print(f"  ✓ Saved processed data to {data_path}")
    
    # Also save the original data format (federalist_papers.json) for reference
    # This was already done in download script, but keep for completeness
    
    return data_path

def tokens_to_ids(tokens, tokenizer):
    """Convert list of tokens to list of token IDs."""
    return [tokenizer.convert_tokens_to_ids(token) for token in tokens]

def train_models(processed_data, tokenizer, device, epochs=10, output_dir="data"):
    """Train all language models."""
    from model_classes import FFNNLM, RNNLM, UALM, ATTNLM, Transformer
    
    print("Converting tokens to IDs...")
    # Convert token lists to ID lists
    train_ids_m = tokens_to_ids(processed_data['train_tokens_m'], tokenizer)
    train_ids_h = tokens_to_ids(processed_data['train_tokens_h'], tokenizer)
    valid_ids_m = tokens_to_ids(processed_data['valid_tokens_m'], tokenizer)
    valid_ids_h = tokens_to_ids(processed_data['valid_tokens_h'], tokenizer)
    
    print(f"Training data - Madison: {len(train_ids_m)} tokens, Hamilton: {len(train_ids_h)} tokens")
    print(f"Validation data - Madison: {len(valid_ids_m)} tokens, Hamilton: {len(valid_ids_h)} tokens")
    
    models_to_train = [
        ("FFNN", "Madison", lambda: FFNNLM(N_GRAM, tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), train_ids_m, valid_ids_m),
        ("FFNN", "Hamilton", lambda: FFNNLM(N_GRAM, tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), train_ids_h, valid_ids_h),
        ("RNN", "Madison", lambda: RNNLM(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), train_ids_m, valid_ids_m),
        ("RNN", "Hamilton", lambda: RNNLM(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), train_ids_h, valid_ids_h),
        ("UALM", "Madison", lambda: UALM(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), train_ids_m, valid_ids_m),
        ("UALM", "Hamilton", lambda: UALM(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), train_ids_h, valid_ids_h),
        ("ATTNLM", "Madison", lambda: ATTNLM(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), train_ids_m, valid_ids_m),
        ("ATTNLM", "Hamilton", lambda: ATTNLM(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), train_ids_h, valid_ids_h),
        ("Transformer", "Madison", lambda: Transformer(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), train_ids_m, valid_ids_m),
        ("Transformer", "Hamilton", lambda: Transformer(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), train_ids_h, valid_ids_h),
    ]
    
    trained_models = {}
    
    for model_type, author, model_factory, train_data, val_data in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_type} Language Model - {author}")
        print(f"{'='*50}")
        
        # Create model
        model = model_factory()
        
        # Train model
        model.train_all(
            train_data, val_data, 
            epochs=epochs, 
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE, 
            sequence_length=SEQUENCE_LENGTH
        )
        
        # Store trained model
        key = f"{model_type.lower()}_lm_{author[0].lower()}"  # e.g., "ffnn_lm_m"
        trained_models[key] = model
        
        print(f"✓ {model_type} {author} model training complete")
    
    return trained_models

def save_models(models, output_dir=DEFAULT_OUTPUT_DIR):
    """Save all trained models to the Models subdirectory."""
    import os
    
    # Create Models subdirectory
    models_dir = os.path.join(output_dir, MODELS_DIR_PATH)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\nSaving trained models to {models_dir}...")
    
    # Model name mapping for lab compatibility
    model_mapping = {
        "ffnn_lm_m": "ffnn_lm_m.pt",
        "ffnn_lm_h": "ffnn_lm_h.pt", 
        "rnn_lm_m": "rnn_lm_m.pt",
        "rnn_lm_h": "rnn_lm_h.pt",
        "ualm_lm_m": "u_attn_lm_m.pt",  # Note: UALM maps to u_attn_lm
        "ualm_lm_h": "u_attn_lm_h.pt",
        "attnlm_lm_m": "attn_lm_m.pt", # Note: ATTNLM maps to attn_lm
        "attnlm_lm_h": "attn_lm_h.pt",
        "transformer_lm_m": "transformer_lm_m.pt",
        "transformer_lm_h": "transformer_lm_h.pt",
    }
    
    for model_key, model in models.items():
        # Load best model state if available
        if hasattr(model, 'best_model') and model.best_model is not None:
            model.load_state_dict(model.best_model)
            print(f"  Loaded best state for {model_key}")
        
        # Get filename
        filename = model_mapping.get(model_key, f"{model_key}.pt")
        filepath = os.path.join(models_dir, filename)
        
        # Save model
        torch.save(model.state_dict(), filepath)
        print(f"  ✓ Saved {model_key} to {filename}")
    
    print("✓ All models saved successfully!")

def print_copy_instructions(output_dir):
    """Print instructions for copying files to the data repo."""
    print(f"\n{'='*60}")
    print("COPY INSTRUCTIONS")
    print(f"{'='*60}")
    print("To update the data repo with the new organized structure, run:")
    print()
    print("# Copy the entire organized Federalist directory:")
    print(f"  cp -r {output_dir}/{FEDERALIST_BASE_DIR} /path/to/your/data-repo/")
    print()
    print("# Or copy individual directories:")
    print(f"  cp -r {output_dir}/{RAW_DIR_PATH} /path/to/your/data-repo/{FEDERALIST_BASE_DIR}/")
    print(f"  cp -r {output_dir}/{TEXT_DIR_PATH} /path/to/your/data-repo/{FEDERALIST_BASE_DIR}/")
    print(f"  cp -r {output_dir}/{MODELS_DIR_PATH} /path/to/your/data-repo/{FEDERALIST_BASE_DIR}/")
    print()
    print("Then commit and push the data repo:")
    print("  cd /path/to/your/data-repo")
    print("  git add Federalist/")
    print("  git commit -m 'Reorganize Federalist data with Raw/Text/Models structure and new HF tokenization'")
    print("  git push")
    print()
    print("DIRECTORY STRUCTURE:")
    print("  Federalist/")
    print("  ├── Raw/           # Original HTML files")
    print("  ├── Text/          # Clean JSON text data")
    print("  └── Models/        # Trained models and tokenizer")
    print()
    print("✓ Your labs can now reference specific subdirectories as needed!")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Build Federalist Papers data and models")
    parser.add_argument("--output-dir", default="data", help="Output directory for generated files")
    parser.add_argument("--no-models", action="store_true", help="Skip model training, only process data")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(SEED)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Step 1: Download raw data
        download_raw_data(args.output_dir)
        
        # Step 2: Load raw data
        raw_data = load_raw_data(args.output_dir)
        if raw_data is None:
            return
        
        # Step 3: Build tokenizer
        tokenizer = build_tokenizer(raw_data)
        
        # Step 4: Process and split data
        processed_data = process_and_split_data(raw_data, tokenizer)
        
        # Step 5: Save processed data and tokenizer
        save_processed_data(processed_data, tokenizer, args.output_dir)
        
        if not args.no_models:
            # Step 6: Train models
            trained_models = train_models(processed_data, tokenizer, device, args.epochs, args.output_dir)
            
            # Step 7: Save models
            save_models(trained_models, args.output_dir)
            
            # Step 8: Print copy instructions
            print_copy_instructions(args.output_dir)
        else:
            print("\nSkipping model training (--no-models flag set)")
        
        print(f"\n✓ Pipeline complete! Files saved to {args.output_dir}/")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()