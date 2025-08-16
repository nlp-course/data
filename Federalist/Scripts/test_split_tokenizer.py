#!/usr/bin/env python3
"""
Test script to debug the Split pre-tokenizer regex pattern.
"""

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.normalizers import Lowercase
from tokenizers.trainers import WordLevelTrainer

def test_split_tokenizer():
    """Test BertPreTokenizer vs custom regex patterns."""
    
    # Test text
    test_text = "Hello, world! This is a test. It has punctuation: commas, periods, and apostrophes like don't."
    
    print("Testing BertPreTokenizer vs custom regex patterns")
    print("=" * 60)
    print(f"Test text: '{test_text}'")
    print()
    
    # Test BertPreTokenizer
    print("BertPreTokenizer (standard approach):")
    print("-" * 40)
    
    try:
        # Create tokenizer with BertPreTokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = BertPreTokenizer()
        
        # Train on simple data
        trainer = WordLevelTrainer(
            min_frequency=1,
            special_tokens=["[UNK]", "[PAD]"]
        )
        tokenizer.train_from_iterator([test_text], trainer=trainer)
        
        # Test tokenization
        encoded = tokenizer.encode(test_text)
        tokens = [tokenizer.id_to_token(token_id) for token_id in encoded.ids]
        
        # Remove special tokens
        tokens = [token for token in tokens if token not in ['[UNK]', '[PAD]']]
        
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"Vocabulary size: {len(tokenizer)}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    
    # Test BertPreTokenizer with lowercase normalizer
    print("BertPreTokenizer with Lowercase normalizer:")
    print("-" * 40)
    
    try:
        # Create tokenizer with BertPreTokenizer and lowercase normalizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = BertPreTokenizer()
        tokenizer.normalizer = Lowercase()
        
        # Train on simple data
        trainer = WordLevelTrainer(
            min_frequency=1,
            special_tokens=["[UNK]", "[PAD]"]
        )
        tokenizer.train_from_iterator([test_text], trainer=trainer)
        
        # Test tokenization
        encoded = tokenizer.encode(test_text)
        tokens = [tokenizer.id_to_token(token_id) for token_id in encoded.ids]
        
        # Remove special tokens
        tokens = [token for token in tokens if token not in ['[UNK]', '[PAD]']]
        
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"Vocabulary size: {len(tokenizer)}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print()

if __name__ == "__main__":
    test_split_tokenizer() 