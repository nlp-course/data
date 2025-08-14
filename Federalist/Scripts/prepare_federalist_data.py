#!/usr/bin/env python3
"""
Federalist Papers Data Preparation Script

This script prepares the Federalist Papers data for use in CS187 labs by:
1. Parsing raw HTML files from Federalist/Raw/
2. Splitting data into train/validation/test sets
3. Building a HuggingFace tokenizer from training data only
4. Tokenizing all data with the trained tokenizer
5. Saving processed data and tokenizer

Usage:
    # From Federalist directory
    python Scripts/prepare_federalist_data.py
"""

import os
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from html.parser import HTMLParser

import torch
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

# Configuration
SEED = 1234
MIN_FREQ = 3
TRAIN_RATIO = 0.9

# Directory structure (when run from Federalist directory)
FEDERALIST_DIR = "."  # Current directory (Federalist/)
RAW_SUBDIR = "Raw"
TEXT_SUBDIR = "Text"
MODELS_SUBDIR = "Models"

# Output files
PROCESSED_DATA_JSON = "federalist_data_processed.json"
TOKENIZER_FILE = "tokenizer.pt"

class HTMLTextExtractor(HTMLParser):
    """Extract text content from HTML, stripping all tags."""
    
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.in_script = False
        self.in_style = False
    
    def handle_starttag(self, tag, attrs):
        if tag.lower() in ['script', 'style']:
            self.in_script = True
    
    def handle_endtag(self, tag):
        if tag.lower() in ['script', 'style']:
            self.in_script = False
    
    def handle_data(self, data):
        if not self.in_script and not self.in_style:
            # Clean up whitespace but preserve paragraph breaks
            cleaned = re.sub(r'\s+', ' ', data.strip())
            if cleaned:
                self.text_parts.append(cleaned)
    
    def get_text(self):
        return ' '.join(self.text_parts)

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)

def extract_metadata_from_filename(filename):
    """Extract paper number, title, and author from filename."""
    # Expected format: "1. General Introduction - Hamilton" (no extension)
    base = filename
    
    # Extract number
    number_match = re.match(r'^(\d+)', base)
    number = number_match.group(1) if number_match else "unknown"
    
    # Split on " - " to get title and author
    if ' - ' in base:
        title_part, author = base.rsplit(' - ', 1)
        # Remove number and period from title
        title = re.sub(r'^\d+\.\s*', '', title_part)
    else:
        title = "Unknown Title"
        author = "Unknown"
    
    return number, title, author

def parse_html_file(filepath):
    """Parse HTML file (with or without .html extension) and extract clean text."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(filepath, 'r', encoding='latin-1') as f:
            html_content = f.read()
    
    extractor = HTMLTextExtractor()
    extractor.feed(html_content)
    return extractor.get_text()

def load_raw_data():
    """Load and parse all HTML files from Raw directory."""
    raw_dir = os.path.join(FEDERALIST_DIR, RAW_SUBDIR)
    
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
    papers = []
    # Look for files that appear to be Federalist Papers (start with number and dot)
    html_files = [f for f in os.listdir(raw_dir) if re.match(r'^\d+\.\s', f)]
    
    if not html_files:
        raise FileNotFoundError(f"No HTML files found in {raw_dir}")
    
    print(f"Found {len(html_files)} HTML files in {raw_dir}")
    
    for filename in sorted(html_files):
        filepath = os.path.join(raw_dir, filename)
        
        # Extract metadata
        number, title, authors = extract_metadata_from_filename(filename)
        
        # Parse HTML and extract text
        text = parse_html_file(filepath)
        
        if text.strip():  # Only include papers with content
            papers.append({
                "number": number,
                "title": title,
                "authors": authors,
                "text": text
            })
            print(f"Loaded paper {number}: {title[:50]}...")
    
    print(f"Successfully loaded {len(papers)} papers")
    return papers

def tokenize_text(text):
    """Simple tokenization - split on whitespace and basic punctuation."""
    # Convert to lowercase and split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    return tokens

def split_data(papers):
    """Split papers into train/validation/test sets."""
    print("Splitting data by author...")
    
    # Separate papers by author
    madison_docs = [doc for doc in papers if 'Madison' in doc['authors'] and 'Hamilton' not in doc['authors']]
    hamilton_docs = [doc for doc in papers if 'Hamilton' in doc['authors'] and 'Madison' not in doc['authors']]
    disputed_docs = [doc for doc in papers if 'Hamilton or Madison' in doc['authors'] or 
                     ('Hamilton' in doc['authors'] and 'Madison' in doc['authors'])]
    other_docs = [doc for doc in papers if doc not in madison_docs + hamilton_docs + disputed_docs]
    
    print(f"Madison papers: {len(madison_docs)}")
    print(f"Hamilton papers: {len(hamilton_docs)}")
    print(f"Disputed papers (test set): {len(disputed_docs)}")
    print(f"Other papers: {len(other_docs)}")
    
    # Shuffle within each author group for random split
    random.shuffle(madison_docs)
    random.shuffle(hamilton_docs)
    
    # Split Madison papers
    madison_train_size = int(math.floor(TRAIN_RATIO * len(madison_docs)))
    madison_train = madison_docs[:madison_train_size]
    madison_valid = madison_docs[madison_train_size:]
    
    # Split Hamilton papers
    hamilton_train_size = int(math.floor(TRAIN_RATIO * len(hamilton_docs)))
    hamilton_train = hamilton_docs[:hamilton_train_size]
    hamilton_valid = hamilton_docs[hamilton_train_size:]
    
    # Combine splits
    train_data = madison_train + hamilton_train
    valid_data = madison_valid + hamilton_valid
    test_data = disputed_docs  # Disputed papers as test set
    
    # Shuffle combined datasets
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)
    
    print(f"Final splits - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    
    return {
        'train': train_data,
        'validation': valid_data,
        'test': test_data
    }

def build_tokenizer(train_data):
    """Build HuggingFace tokenizer from training data only."""
    print("Building tokenizer from training data...")
    
    # Collect all text from training set only
    all_tokens = []
    for doc in train_data:
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
    
    # Ensure proper attribute availability after serialization
    hf_tokenizer.split_special_tokens = False
    
    print(f"Final tokenizer vocabulary size: {len(hf_tokenizer)}")
    return hf_tokenizer

def tokenize_data(splits, tokenizer):
    """Tokenize all data using the trained tokenizer."""
    print("Tokenizing all data with trained tokenizer...")
    
    tokenized_splits = {}
    
    for split_name, documents in splits.items():
        tokenized_docs = []
        
        for doc in documents:
            # Use the trained tokenizer to get consistent token IDs
            tokens = tokenizer.tokenize(doc['text'])
            
            tokenized_doc = {
                'number': doc['number'],
                'title': doc['title'],
                'authors': doc['authors'],
                'tokens': tokens,  # Keep as tokens for lab compatibility
                'text': doc['text']  # Keep original text too
            }
            tokenized_docs.append(tokenized_doc)
        
        tokenized_splits[split_name] = tokenized_docs
        print(f"{split_name}: {len(tokenized_docs)} documents")
    
    return tokenized_splits

def save_data(tokenized_splits, tokenizer):
    """Save processed data and tokenizer."""
    # Create output directories if needed
    text_dir = os.path.join(FEDERALIST_DIR, TEXT_SUBDIR)
    models_dir = os.path.join(FEDERALIST_DIR, MODELS_SUBDIR)
    
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save processed data
    output_path = os.path.join(text_dir, PROCESSED_DATA_JSON)
    with open(output_path, 'w') as f:
        json.dump(tokenized_splits, f, indent=2)
    print(f"Saved processed data to: {output_path}")
    
    # Save tokenizer
    tokenizer_path = os.path.join(text_dir, TOKENIZER_FILE)
    torch.save(tokenizer, tokenizer_path)
    print(f"Saved tokenizer to: {tokenizer_path}")
    
    return output_path, tokenizer_path

def print_summary(tokenized_splits, tokenizer):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("DATA PREPARATION SUMMARY")
    print("="*60)
    
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    
    total_papers = sum(len(docs) for docs in tokenized_splits.values())
    print(f"Total papers processed: {total_papers}")
    
    for split_name, documents in tokenized_splits.items():
        total_tokens = sum(len(doc['tokens']) for doc in documents)
        avg_tokens = total_tokens / len(documents) if documents else 0
        print(f"{split_name.capitalize()}: {len(documents)} papers, {total_tokens} tokens (avg: {avg_tokens:.1f})")
    
    print("\nFiles created:")
    print(f"- {FEDERALIST_DIR}/{TEXT_SUBDIR}/{PROCESSED_DATA_JSON}")
    print(f"- {FEDERALIST_DIR}/{TEXT_SUBDIR}/{TOKENIZER_FILE}")
    
    print(f"\nNext step: Run Scripts/train_federalist_models.py to train language models")

def main():
    """Main execution function."""
    print("Federalist Papers Data Preparation")
    print("="*50)
    
    # Set random seed
    set_seed(SEED)
    print(f"Random seed set to: {SEED}")
    
    # Load raw data
    papers = load_raw_data()
    
    # Split data
    splits = split_data(papers)
    
    # Build tokenizer from training data only
    tokenizer = build_tokenizer(splits['train'])
    
    # Tokenize all data
    tokenized_splits = tokenize_data(splits, tokenizer)
    
    # Save results
    save_data(tokenized_splits, tokenizer)
    
    # Print summary
    print_summary(tokenized_splits, tokenizer)
    
    print("\n✓ Data preparation completed successfully!")

if __name__ == "__main__":
    main()