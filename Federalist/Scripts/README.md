# Federalist Papers Data Processing Scripts

This directory contains scripts for processing the Federalist Papers data used in CS187 labs.

## Scripts

### `download_raw_data_from_yale.sh`
Downloads raw HTML content of all 85 Federalist Papers from Yale Law School Avalon Project.
- **Usage**: `bash Scripts/download_raw_data_from_yale.sh` (from Federalist directory)
- **Output**: HTML files in `Raw/` directory
- **When to use**: Initial setup or when you need fresh raw data

### `prepare_federalist_data.py`
Main data preparation script that processes raw HTML files into tokenized data.
- **Usage**: `python Scripts/prepare_federalist_data.py` (from Federalist directory)
- **Input**: HTML files from `Raw/` directory
- **Output**: Processed data and tokenizer in `Text/` directory
- **When to use**: After downloading raw data or when you need to regenerate processed data

### `train_federalist_models.py`
Trains language models on the processed data.
- **Usage**: `python Scripts/train_federalist_models.py` (from Federalist directory)
- **Input**: Processed data and tokenizer from `Text/` directory
- **Output**: Trained models in `Models/` directory
- **When to use**: After data preparation, when you need to train or retrain models

### `build_federalist_data.py` (Legacy)
Original combined pipeline script - now replaced by the separate scripts above.

**What it does:**
1. Downloads raw Federalist Papers data from the repo
2. Builds a HuggingFace tokenizer from the raw text
3. Tokenizes and splits data into train/validation/test sets
4. Trains all language models (FFNN, RNN, UALM, ATTNLM, Transformer)
5. Saves everything to the appropriate directories

**Usage:**
```bash
# From the Data repo root directory
cd /path/to/CS187/Data
python Federalist/Scripts/build_federalist_data.py
```

**Output:**
- `Federalist/Text/federalist_data_processed.json` - Processed and split data
- `Federalist/Text/tokenizer.pt` - HuggingFace tokenizer
- `Federalist/Models/*.pt` - Trained model files

### `model_classes.py`
Contains the language model class definitions used by the main pipeline.

### `download_federalist_from_repo.py`
Downloads and performs initial processing of raw Federalist Papers data.

## Requirements

Make sure you have the CS187 conda environment activated:
```bash
conda activate cs187-env
```

The script requires PyTorch, transformers, tokenizers, and other packages included in the course environment.

## Directory Structure

After running the pipeline, the data will be organized as:
```
Federalist/
├── Raw/                    # Raw HTML/text files
├── Text/                   # Processed data and tokenizer
│   ├── federalist_data_processed.json
│   └── tokenizer.pt
├── Models/                 # Trained language models
│   ├── ffnn_lm_h.pt
│   ├── ffnn_lm_m.pt
│   ├── rnn_lm_h.pt
│   ├── rnn_lm_m.pt
│   ├── u_attn_lm_h.pt
│   ├── u_attn_lm_m.pt
│   ├── attn_lm_h.pt
│   ├── attn_lm_m.pt
│   ├── transformer_lm_h.pt
│   └── transformer_lm_m.pt
└── Scripts/                # This directory
```