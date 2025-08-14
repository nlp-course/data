#!/usr/bin/env python3
"""
Federalist Papers Model Training Script

This script trains language models on the processed Federalist Papers data by:
1. Loading processed data and tokenizer from Text directory
2. Training multiple model types with different configurations
3. Saving trained models to Models directory

Usage:
    # From Federalist directory
    python Scripts/train_federalist_models.py [--epochs N] [--no-models]
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import model classes (assumes model_classes.py is in same directory)
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_classes import FFNNLM, RNNLM, UALM, ATTNLM, Transformer

# Configuration
SEED = 1234
DEFAULT_EPOCHS = 10

# Directory structure (when run from Federalist directory)
FEDERALIST_DIR = "."  # Current directory (Federalist/)
TEXT_SUBDIR = "Text"
MODELS_SUBDIR = "Models"

# Input files
PROCESSED_DATA_JSON = "federalist_data_processed.json"
TOKENIZER_FILE = "tokenizer.pt"

# Model hyperparameters
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
N_GRAM = 5
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
SEQUENCE_LENGTH = 35

# Model-specific epochs
MODEL_EPOCHS = {
    'FFNN': 10,
    'RNN': 10,
    'UALM': 8,
    'ATTNLM': 8,
    'Transformer': 6
}

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)

def tokens_to_ids(tokens, tokenizer):
    """Convert token list to ID list using tokenizer."""
    return [tokenizer.convert_tokens_to_ids(token) for token in tokens]

def load_processed_data():
    """Load processed data and tokenizer."""
    text_dir = os.path.join(FEDERALIST_DIR, TEXT_SUBDIR)
    
    # Load processed data
    data_path = os.path.join(text_dir, PROCESSED_DATA_JSON)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found: {data_path}")
    
    with open(data_path, 'r') as f:
        splits = json.load(f)
    
    # Load tokenizer
    tokenizer_path = os.path.join(text_dir, TOKENIZER_FILE)
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    tokenizer = torch.load(tokenizer_path, weights_only=False)
    
    print(f"Loaded data splits: {list(splits.keys())}")
    print(f"Loaded tokenizer with vocabulary size: {len(tokenizer)}")
    
    return splits, tokenizer

def prepare_training_data(splits, tokenizer):
    """Convert tokenized data to ID sequences for training."""
    print("Converting tokens to IDs...")
    
    # Convert token lists to ID lists
    train_data = {}
    valid_data = {}
    
    # Separate Madison and Hamilton data for training separate models
    for split_name in ['train', 'validation']:
        documents = splits[split_name]
        
        madison_docs = [doc for doc in documents if 'Madison' in doc['authors'] and 'Hamilton' not in doc['authors']]
        hamilton_docs = [doc for doc in documents if 'Hamilton' in doc['authors'] and 'Madison' not in doc['authors']]
        
        # Combine all tokens for each author
        madison_tokens = []
        hamilton_tokens = []
        
        for doc in madison_docs:
            madison_tokens.extend(doc['tokens'])
        
        for doc in hamilton_docs:
            hamilton_tokens.extend(doc['tokens'])
        
        # Convert to IDs
        madison_ids = tokens_to_ids(madison_tokens, tokenizer)
        hamilton_ids = tokens_to_ids(hamilton_tokens, tokenizer)
        
        if split_name == 'train':
            train_data['madison'] = madison_ids
            train_data['hamilton'] = hamilton_ids
        else:
            valid_data['madison'] = madison_ids
            valid_data['hamilton'] = hamilton_ids
    
    print(f"Training data - Madison: {len(train_data['madison'])} tokens, Hamilton: {len(train_data['hamilton'])} tokens")
    print(f"Validation data - Madison: {len(valid_data['madison'])} tokens, Hamilton: {len(valid_data['hamilton'])} tokens")
    
    return train_data, valid_data

def train_models(train_data, valid_data, tokenizer, device, epochs_override=None):
    """Train all language models."""
    print(f"Training models on device: {device}")
    
    # Models to train: (model_type, author, model_factory, train_data, val_data)
    models_to_train = [
        ("FFNN", "Madison", lambda: FFNNLM(N_GRAM, tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), 
         train_data['madison'], valid_data['madison']),
        ("FFNN", "Hamilton", lambda: FFNNLM(N_GRAM, tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), 
         train_data['hamilton'], valid_data['hamilton']),
        ("RNN", "Madison", lambda: RNNLM(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), 
         train_data['madison'], valid_data['madison']),
        ("RNN", "Hamilton", lambda: RNNLM(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), 
         train_data['hamilton'], valid_data['hamilton']),
        ("UALM", "Madison", lambda: UALM(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), 
         train_data['madison'], valid_data['madison']),
        ("UALM", "Hamilton", lambda: UALM(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), 
         train_data['hamilton'], valid_data['hamilton']),
        ("ATTNLM", "Madison", lambda: ATTNLM(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), 
         train_data['madison'], valid_data['madison']),
        ("ATTNLM", "Hamilton", lambda: ATTNLM(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), 
         train_data['hamilton'], valid_data['hamilton']),
        ("Transformer", "Madison", lambda: Transformer(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), 
         train_data['madison'], valid_data['madison']),
        ("Transformer", "Hamilton", lambda: Transformer(tokenizer, EMBEDDING_SIZE, HIDDEN_SIZE).to(device), 
         train_data['hamilton'], valid_data['hamilton']),
    ]
    
    trained_models = {}
    
    for model_type, author, model_factory, train_data_author, val_data_author in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_type} Language Model - {author}")
        print(f"{'='*60}")
        
        # Determine epochs for this model type
        if epochs_override:
            epochs = epochs_override
        else:
            epochs = MODEL_EPOCHS.get(model_type, DEFAULT_EPOCHS)
        
        print(f"Training for {epochs} epochs")
        
        # Create model
        model = model_factory()
        
        # Train model
        try:
            model.train_all(
                train_data_author, val_data_author, 
                epochs=epochs, 
                learning_rate=LEARNING_RATE,
                batch_size=BATCH_SIZE, 
                sequence_length=SEQUENCE_LENGTH
            )
            
            # Store trained model
            key = f"{model_type.lower()}_lm_{author[0].lower()}"  # e.g., "ffnn_lm_m"
            trained_models[key] = model
            
            print(f"✓ {model_type} {author} model training complete")
            
        except Exception as e:
            print(f"✗ Error training {model_type} {author} model: {e}")
            continue
    
    return trained_models

def save_models(models):
    """Save all trained models to the Models directory."""
    models_dir = os.path.join(FEDERALIST_DIR, MODELS_SUBDIR)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\nSaving models to {models_dir}/")
    
    saved_files = []
    for model_name, model in models.items():
        filename = f"{model_name}.pt"
        filepath = os.path.join(models_dir, filename)
        
        try:
            torch.save(model.state_dict(), filepath)
            saved_files.append(filename)
            print(f"✓ Saved {filename}")
        except Exception as e:
            print(f"✗ Error saving {filename}: {e}")
    
    return saved_files

def print_summary(trained_models, saved_files):
    """Print training summary."""
    print("\n" + "="*60)
    print("MODEL TRAINING SUMMARY")
    print("="*60)
    
    print(f"Models trained: {len(trained_models)}")
    print(f"Models saved: {len(saved_files)}")
    
    print("\nTrained models:")
    for model_name in sorted(trained_models.keys()):
        print(f"  - {model_name}")
    
    print(f"\nSaved files in {FEDERALIST_DIR}/{MODELS_SUBDIR}/:")
    for filename in sorted(saved_files):
        print(f"  - {filename}")
    
    print("\nThese models can now be used in labs 2-3, 2-6, and 2-7!")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train Federalist Papers language models")
    parser.add_argument("--epochs", type=int, default=None, 
                       help="Override default epochs for all models")
    parser.add_argument("--no-models", action="store_true",
                       help="Skip model training (data preparation only)")
    
    args = parser.parse_args()
    
    print("Federalist Papers Model Training")
    print("="*50)
    
    # Set random seed
    set_seed(SEED)
    print(f"Random seed set to: {SEED}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.no_models:
        print("Skipping model training (--no-models specified)")
        return
    
    # Load processed data
    splits, tokenizer = load_processed_data()
    
    # Prepare training data
    train_data, valid_data = prepare_training_data(splits, tokenizer)
    
    # Train models
    trained_models = train_models(train_data, valid_data, tokenizer, device, args.epochs)
    
    # Save models
    saved_files = save_models(trained_models)
    
    # Print summary
    print_summary(trained_models, saved_files)
    
    print("\n✓ Model training completed successfully!")

if __name__ == "__main__":
    main()