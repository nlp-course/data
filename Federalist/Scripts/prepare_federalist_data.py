#!/usr/bin/env python3
"""
Federalist Papers Data Preparation Script

This script prepares the Federalist Papers data for use in CS187 labs by:
1. Parsing the Project Gutenberg Federalist Papers file from Federalist/Raw/
2. Using authoritative author and title data from Library of Congress
3. Splitting data into train/validation/test sets
4. Building a HuggingFace tokenizer from training data only
5. Tokenizing all data with the trained tokenizer
6. Saving processed data and tokenizer

Usage:
    # From Federalist directory
    python Scripts/prepare_federalist_data.py

Prerequisites:
    - Run download_federalist_from_gutenberg.py first to download the Gutenberg file
"""

import os
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from tokenizers.normalizers import Lowercase

# Configuration
SEED = 1234
MIN_FREQ = 3
TRAIN_RATIO = 0.9

# Keywords for count analysis (used in lab 1-2)
KEYWORDS = ['on', 'upon', 'there', 'whilst']

# Directory structure (when run from Federalist directory)
FEDERALIST_DIR = "."  # Current directory (Federalist/)
RAW_SUBDIR = "Raw"
TEXT_SUBDIR = "Text"
MODELS_SUBDIR = "Models"

# Output files
PROCESSED_DATA_JSON = "federalist_data_processed.json"
PROCESSED_DATA_FOUR_COUNTS_JSON = "federalist_data_four_counts.json"
TOKENIZER_FILE = "tokenizer.pt"

# Authoritative author and title data extracted from Library of Congress web page
# https://guides.loc.gov/federalist-papers/full-text
FEDERALIST_DATA = {
    1: {"title": "General Introduction", "author": "Hamilton"},
    2: {"title": "Concerning Dangers from Foreign Force and Influence", "author": "Jay"},
    3: {"title": "The Same Subject Continued: Concerning Dangers from Foreign Force and Influence", "author": "Jay"},
    4: {"title": "The Same Subject Continued: Concerning Dangers from Foreign Force and Influence", "author": "Jay"},
    5: {"title": "The Same Subject Continued: Concerning Dangers from Foreign Force and Influence", "author": "Jay"},
    6: {"title": "Concerning Dangers from Dissensions Between the States", "author": "Hamilton"},
    7: {"title": "The Same Subject Continued: Concerning Dangers from Dissensions Between the States", "author": "Hamilton"},
    8: {"title": "The Consequences of Hostilities Between the States", "author": "Hamilton"},
    9: {"title": "The Union as a Safeguard Against Domestic Faction and Insurrection", "author": "Hamilton"},
    10: {"title": "The Same Subject Continued: The Union as a Safeguard Against Domestic Faction and Insurrection", "author": "Madison"},
    11: {"title": "The Utility of the Union in Respect to Commercial Relations and a Navy", "author": "Hamilton"},
    12: {"title": "The Utility of the Union in Respect to Revenue", "author": "Hamilton"},
    13: {"title": "Advantage of the Union in Respect to Economy in Government", "author": "Hamilton"},
    14: {"title": "Objections to the Proposed Constitution from Extent of Territory Answered", "author": "Madison"},
    15: {"title": "The Insufficiency of the Present Confederation to Preserve the Union", "author": "Hamilton"},
    16: {"title": "The Same Subject Continued: The Insufficiency of the Present Confederation to Preserve the Union", "author": "Hamilton"},
    17: {"title": "The Same Subject Continued: The Insufficiency of the Present Confederation to Preserve the Union", "author": "Hamilton"},
    18: {"title": "The Same Subject Continued: The Insufficiency of the Present Confederation to Preserve the Union", "author": "Madison and Hamilton"},
    19: {"title": "The Same Subject Continued: The Insufficiency of the Present Confederation to Preserve the Union", "author": "Madison and Hamilton"},
    20: {"title": "The Same Subject Continued: The Insufficiency of the Present Confederation to Preserve the Union", "author": "Madison and Hamilton"},
    21: {"title": "Other Defects of the Present Confederation", "author": "Hamilton"},
    22: {"title": "The Same Subject Continued: Other Defects of the Present Confederation", "author": "Hamilton"},
    23: {"title": "The Necessity of a Government as Energetic as the One Proposed to the Attainment of This Object", "author": "Hamilton"},
    24: {"title": "The Powers Necessary to the Common Defense Further Considered", "author": "Hamilton"},
    25: {"title": "The Same Subject Continued: The Powers Necessary to the Common Defense Further Considered", "author": "Hamilton"},
    26: {"title": "The Idea of Restraining the Legislative Authority in Regard to the Common Defense Considered", "author": "Hamilton"},
    27: {"title": "The Same Subject Continued: The Idea of Restraining the Legislative Authority in Regard to the Common Defense Considered", "author": "Hamilton"},
    28: {"title": "The Same Subject Continued: The Idea of Restraining the Legislative Authority in Regard to the Common Defense Considered", "author": "Hamilton"},
    29: {"title": "Concerning the Militia", "author": "Hamilton"},
    30: {"title": "Concerning the General Power of Taxation", "author": "Hamilton"},
    31: {"title": "The Same Subject Continued: Concerning the General Power of Taxation", "author": "Hamilton"},
    32: {"title": "The Same Subject Continued: Concerning the General Power of Taxation", "author": "Hamilton"},
    33: {"title": "The Same Subject Continued: Concerning the General Power of Taxation", "author": "Hamilton"},
    34: {"title": "The Same Subject Continued: Concerning the General Power of Taxation", "author": "Hamilton"},
    35: {"title": "The Same Subject Continued: Concerning the General Power of Taxation", "author": "Hamilton"},
    36: {"title": "The Same Subject Continued: Concerning the General Power of Taxation", "author": "Hamilton"},
    37: {"title": "Concerning the Difficulties of the Convention in Devising a Proper Form of Government", "author": "Madison"},
    38: {"title": "The Same Subject Continued, and the Incoherence of the Objections to the New Plan Exposed", "author": "Madison"},
    39: {"title": "The Conformity of the Plan to Republican Principles", "author": "Madison"},
    40: {"title": "On the Powers of the Convention to Form a Mixed Government Examined and Sustained", "author": "Madison"},
    41: {"title": "General View of the Powers Conferred by The Constitution", "author": "Madison"},
    42: {"title": "The Powers Conferred by the Constitution Further Considered", "author": "Madison"},
    43: {"title": "The Same Subject Continued: The Powers Conferred by the Constitution Further Considered", "author": "Madison"},
    44: {"title": "Restrictions on the Authority of the Several States", "author": "Madison"},
    45: {"title": "The Alleged Danger From the Powers of the Union to the State Governments Considered", "author": "Madison"},
    46: {"title": "The Influence of the State and Federal Governments Compared", "author": "Madison"},
    47: {"title": "The Particular Structure of the New Government and the Distribution of Power Among Its Different Parts", "author": "Madison"},
    48: {"title": "These Departments Should Not Be So Far Separated as to Have No Constitutional Control Over Each Other", "author": "Madison"},
    49: {"title": "Method of Guarding Against the Encroachments of Any One Department of Government by Appealing to the People Through a Convention", "author": "Hamilton or Madison"},
    50: {"title": "Periodical Appeals to the People Considered", "author": "Hamilton or Madison"},
    51: {"title": "The Structure of the Government Must Furnish the Proper Checks and Balances Between the Different Departments", "author": "Hamilton or Madison"},
    52: {"title": "The House of Representatives", "author": "Hamilton or Madison"},
    53: {"title": "The Same Subject Continued: The House of Representatives", "author": "Hamilton or Madison"},
    54: {"title": "The Apportionment of Members Among the States", "author": "Hamilton or Madison"},
    55: {"title": "The Total Number of the House of Representatives", "author": "Hamilton or Madison"},
    56: {"title": "The Same Subject Continued: The Total Number of the House of Representatives", "author": "Hamilton or Madison"},
    57: {"title": "The Alleged Tendency of the New Plan to Elevate the Few at the Expense of the Many Considered in Connection with Representation", "author": "Hamilton or Madison"},
    58: {"title": "Objection That The Number of Members Will Not Be Augmented as the Progress of Population Demands Considered", "author": "Madison"},
    59: {"title": "Concerning the Power of Congress to Regulate the Election of Members", "author": "Hamilton"},
    60: {"title": "The Same Subject Continued: Concerning the Power of Congress to Regulate the Election of Members", "author": "Hamilton"},
    61: {"title": "The Same Subject Continued: Concerning the Power of Congress to Regulate the Election of Members", "author": "Hamilton"},
    62: {"title": "The Senate", "author": "Hamilton or Madison"},
    63: {"title": "The Senate Continued", "author": "Hamilton or Madison"},
    64: {"title": "The Powers of the Senate", "author": "Jay"},
    65: {"title": "The Powers of the Senate Continued", "author": "Hamilton"},
    66: {"title": "Objections to the Power of the Senate To Set as a Court for Impeachments Further Considered", "author": "Hamilton"},
    67: {"title": "The Executive Department", "author": "Hamilton"},
    68: {"title": "The Mode of Electing the President", "author": "Hamilton"},
    69: {"title": "The Real Character of the Executive", "author": "Hamilton"},
    70: {"title": "The Executive Department Further Considered", "author": "Hamilton"},
    71: {"title": "The Duration in Office of the Executive", "author": "Hamilton"},
    72: {"title": "The Same Subject Continued, and Re-Eligibility of the Executive Considered", "author": "Hamilton"},
    73: {"title": "The Provision For The Support of the Executive, and the Veto Power", "author": "Hamilton"},
    74: {"title": "The Command of the Military and Naval Forces, and the Pardoning Power of the Executive", "author": "Hamilton"},
    75: {"title": "The Treaty-Making Power of the Executive", "author": "Hamilton"},
    76: {"title": "The Appointing Power of the Executive", "author": "Hamilton"},
    77: {"title": "The Appointing Power Continued and Other Powers of the Executive Considered", "author": "Hamilton"},
    78: {"title": "The Judiciary Department", "author": "Hamilton"},
    79: {"title": "The Judiciary Continued", "author": "Hamilton"},
    80: {"title": "The Powers of the Judiciary", "author": "Hamilton"},
    81: {"title": "The Judiciary Continued, and the Distribution of the Judicial Authority", "author": "Hamilton"},
    82: {"title": "The Judiciary Continued", "author": "Hamilton"},
    83: {"title": "The Judiciary Continued in Relation to Trial by Jury", "author": "Hamilton"},
    84: {"title": "Certain General and Miscellaneous Objections to the Constitution Considered and Answered", "author": "Hamilton"},
    85: {"title": "Concluding Remarks", "author": "Hamilton"}
}

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)

def parse_gutenberg_file(filepath):
    """Parse the Gutenberg Federalist Papers file and extract individual papers."""
    print(f"Parsing Gutenberg file: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        print("Please run download_federalist_from_gutenberg.py first")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    print(f"File size: {len(content):,} characters")
    
    # Find the start of the actual content (after Gutenberg header)
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK THE FEDERALIST PAPERS ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK THE FEDERALIST PAPERS ***"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("Error: Could not find Gutenberg markers in file")
        return None
    
    # Extract the main content
    content = content[start_idx + len(start_marker):end_idx].strip()
    print(f"Content after removing headers: {len(content):,} characters")
    
    # Split into individual papers
    papers = []
    
    # Pattern to match Federalist paper headers - look for "FEDERALIST No. X"
    paper_pattern = r'FEDERALIST\s+No\.\s+(\d+)'
    
    matches = list(re.finditer(paper_pattern, content))
    
    print(f"Found {len(matches)} potential papers")
    
    for i, match in enumerate(matches):
        number = int(match.group(1))
        
        # Find the start of this paper's content
        paper_start = match.start()
        
        # Find the end of this paper (start of next paper or end of file)
        if i < len(matches) - 1:
            paper_end = matches[i + 1].start()
        else:
            paper_end = len(content)
        
        # Extract the full paper content
        paper_content = content[paper_start:paper_end].strip()
        
        # Parse the paper content
        paper_data = parse_individual_paper(paper_content, number)
        
        if paper_data and paper_data['text'].strip():
            papers.append(paper_data)
            print(f"Paper {number}: {paper_data['title'][:50]}... (Author: {paper_data['authors']})")
    
    print(f"Successfully parsed {len(papers)} papers")
    return papers

def parse_individual_paper(paper_content, number):
    """Parse an individual paper's content to extract text using authoritative metadata."""
    lines = paper_content.split('\n')
    
    # Use authoritative title and author from Library of Congress data
    if number not in FEDERALIST_DATA:
        print(f"Warning: No authoritative data for paper {number}")
        return None
    
    title = FEDERALIST_DATA[number]["title"]
    author = FEDERALIST_DATA[number]["author"]
    
    # Find the start of the text (after "To the People of the State of New York:")
    text_start = None
    for i, line in enumerate(lines):
        if "To the People of the State of New York" in line:
            text_start = i + 1
            break
    
    if text_start is None:
        print(f"Warning: Could not find text start for paper {number}")
        return None
    
    # Get the text content
    text_lines = lines[text_start:]
    text = '\n'.join(text_lines).strip()
    
    # Clean the text
    text = clean_text(text)
    
    return {
        "number": number,
        "title": title,
        "authors": author,  # Use author directly, not wrapped in list
        "text": text
    }

def clean_text(text):
    """Clean up the text content."""
    # Remove extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove common Gutenberg artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
    
    # Clean up line breaks
    text = text.strip()
    
    return text

def load_raw_data():
    """Load and parse the Federalist Papers from the Gutenberg file."""
    raw_dir = os.path.join(FEDERALIST_DIR, RAW_SUBDIR)
    
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    
    # Look for the Gutenberg file
    gutenberg_file = os.path.join(raw_dir, "federalist_papers_gutenberg.txt")
    
    if not os.path.exists(gutenberg_file):
        raise FileNotFoundError(
            f"Gutenberg file not found: {gutenberg_file}\n"
            "Please run download_federalist_from_gutenberg.py first"
        )
    
    # Parse the Gutenberg file
    papers = parse_gutenberg_file(gutenberg_file)
    
    if not papers:
        raise RuntimeError("Failed to parse Gutenberg file")
    
    return papers



def calculate_keyword_counts(text):
    """Calculate counts for the four keywords using torchtext method to match original data."""
    try:
        import torchtext
        # Use the same tokenizer as the original script
        tokenizer = torchtext.data.get_tokenizer("basic_english")
        tokens = tokenizer(text)
        
        # Count each keyword
        counts = []
        for keyword in KEYWORDS:
            count = tokens.count(keyword)
            counts.append(count)
        
        return counts
    except ImportError:
        # Fallback to regex method if torchtext not available
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        counts = []
        for keyword in KEYWORDS:
            count = tokens.count(keyword)
            counts.append(count)
        return counts

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
    others_data = other_docs  # Other papers (Jay) as separate split
    
    # Shuffle combined datasets
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)
    random.shuffle(others_data)
    
    print(f"Final splits - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}, Others: {len(others_data)}")
    
    return {
        'train': train_data,
        'validation': valid_data,
        'test': test_data,
        'others': others_data
    }

def build_tokenizer(train_data):
    """Build HuggingFace tokenizer from training data only."""
    print("Building tokenizer from training data...")
    
    # Collect all text from training set only
    all_text = []
    for doc in train_data:
        all_text.append(doc['text'])
    
    # Create a tokenizer using BertPreTokenizer for clean, standard tokenization
    word_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    word_tokenizer.pre_tokenizer = BertPreTokenizer()
    word_tokenizer.normalizer = Lowercase()
    
    # Train the tokenizer on the training data with more inclusive settings
    word_tokenizer.train_from_iterator(all_text, trainer=WordLevelTrainer(
        min_frequency=1,  # Include all words, even if they appear only once
        special_tokens=["[UNK]", "[PAD]"]
    ))
    
    # Create PreTrainedTokenizerFast (same as in labs)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=word_tokenizer,
        pad_token='[PAD]',
        unk_token='[UNK]'
    )
    
    # Ensure proper attribute availability after serialization
    hf_tokenizer.split_special_tokens = False
    
    # Test the tokenizer to make sure it works correctly
    test_text = "Hello, world! This is a test, including internal punctuation like \"don't\"."
    test_tokens = hf_tokenizer.tokenize(test_text)
    print(f"Test tokenization: '{test_text}' -> {test_tokens}")
    
    print(f"Final tokenizer vocabulary size: {len(hf_tokenizer)}")
    return hf_tokenizer

def tokenize_data(splits, tokenizer):
    """Tokenize all data using the trained tokenizer."""
    print("Tokenizing all data with trained tokenizer...")
    
    tokenized_splits = {}
    
    for split_name, documents in splits.items():
        tokenized_docs = []
        
        for doc in documents:
            # Use the trained tokenizer to get consistent tokenization
            tokens = tokenizer.tokenize(doc['text'])
            
            # Calculate keyword counts for lab 1-2 using torchtext method
            counts = calculate_keyword_counts(doc['text'])
            
            tokenized_doc = {
                'number': doc['number'],
                'title': doc['title'],
                'authors': doc['authors'],
                'tokens': tokens,  # Keep as tokens for lab compatibility
                'text': doc['text'],  # Keep original text too
                'counts': counts  # Add counts for lab 1-2
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
    
    # Save processed data with splits (authors already in correct format from FEDERALIST_DATA)
    processed_data_path = os.path.join(text_dir, PROCESSED_DATA_JSON)
    with open(processed_data_path, 'w') as f:
        json.dump(tokenized_splits, f, indent=2)
    print(f"Saved processed data to: {processed_data_path}")
    
    # Save lab 1-2 compatible data (flat structure, no text/tokens, no splits)
    lab1_2_data = []
    for split_name, documents in tokenized_splits.items():
        for doc in documents:
            # Authors are already in correct format from FEDERALIST_DATA
            lab1_2_doc = {
                'number': doc['number'],
                'title': doc['title'],
                'authors': doc['authors'],
                'counts': doc['counts']
            }
            lab1_2_data.append(lab1_2_doc)
    
    # Sort by paper number for consistency
    lab1_2_data.sort(key=lambda x: x['number'])
    
    lab1_2_path = os.path.join(text_dir, PROCESSED_DATA_FOUR_COUNTS_JSON)
    with open(lab1_2_path, 'w') as f:
        json.dump(lab1_2_data, f, indent=2)
    print(f"Saved lab 1-2 compatible data to: {lab1_2_path}")
    
    # Save tokenizer
    tokenizer_path = os.path.join(text_dir, TOKENIZER_FILE)
    torch.save(tokenizer, tokenizer_path)
    print(f"Saved tokenizer to: {tokenizer_path}")
    
    return processed_data_path, tokenizer_path

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