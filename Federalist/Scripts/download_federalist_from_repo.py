#!/usr/bin/env python3
"""
Download Federalist Papers from the nlp-course data repository.

This downloads the individual papers that are already cleanly separated
and properly titled in the data repo. It extracts the raw text from the HTML files.
"""

import os
import json
import urllib.request
import urllib.parse
import re
from html.parser import HTMLParser

# Path constants
GITHUB_API_BASE = "https://api.github.com/repos/nlp-course/data/contents"
GITHUB_BRANCH = "develop"  # Using develop branch temporarily
FEDERALIST_DATA_URL = f"https://raw.githubusercontent.com/nlp-course/data/{GITHUB_BRANCH}/Federalist/federalist_data_raw2.json"

# Local directory structure - when run from Data repo root
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
            self.in_style = True
    
    def handle_endtag(self, tag):
        if tag.lower() in ['script', 'style']:
            self.in_script = False
            self.in_style = False
        elif tag.lower() in ['p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            # Add newlines for block elements
            self.text_parts.append('\n')
    
    def handle_data(self, data):
        if not (self.in_script or self.in_style):
            # Clean up whitespace but preserve line breaks
            cleaned = re.sub(r'[ \t]+', ' ', data)
            self.text_parts.append(cleaned)
    
    def get_text(self):
        # Join all text parts and clean up extra whitespace
        text = ''.join(self.text_parts)
        # Normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)
        return text.strip()

def clean_html_content(content):
    """Remove HTML tags and extract clean text."""
    if not content.strip():
        return ""
    
    # Check if content looks like HTML
    if '<' in content and '>' in content:
        extractor = HTMLTextExtractor()
        extractor.feed(content)
        cleaned = extractor.get_text()
        
        # Additional cleanup
        cleaned = re.sub(r'&nbsp;', ' ', cleaned)
        cleaned = re.sub(r'&amp;', '&', cleaned)
        cleaned = re.sub(r'&lt;', '<', cleaned)
        cleaned = re.sub(r'&gt;', '>', cleaned)
        cleaned = re.sub(r'&quot;', '"', cleaned)
        cleaned = re.sub(r'&#39;', "'", cleaned)
        
        return cleaned
    else:
        # Already plain text, just clean up
        return content.strip()

def download_federalist_raw_json():
    """Download the existing federalist_data_raw2.json file from the data repo."""
    print("Downloading federalist_data_raw2.json from nlp-course data repo...")
    try:
        with urllib.request.urlopen(FEDERALIST_DATA_URL) as response:
            data = json.load(response)
        
        print(f"✓ Downloaded data with {len(data)} papers")
        return data
        
    except Exception as e:
        print(f"Error downloading federalist data: {e}")
        return None

def extract_paper_number(filename):
    """Extract paper number from filename."""
    match = re.match(r'^(\d+)\.', filename)
    return int(match.group(1)) if match else 999

def parse_filename(filename):
    """Parse filename to extract number, title, and author."""
    # Pattern: "Number. Title - Author"
    match = re.match(r'^(\d+)\.\s*(.+?)\s*-\s*(.+)$', filename)
    
    if match:
        number = int(match.group(1))
        title = match.group(2).strip()
        author = match.group(3).strip()
        return number, title, author
    else:
        # Fallback parsing
        parts = filename.split('-')
        if len(parts) >= 2:
            author = parts[-1].strip()
            title_part = '-'.join(parts[:-1]).strip()
            # Extract number
            number_match = re.match(r'^(\d+)', title_part)
            number = int(number_match.group(1)) if number_match else 0
            title = re.sub(r'^\d+\.\s*', '', title_part).strip()
            return number, title, author
        else:
            return 0, filename, "Unknown"

def process_raw_data(raw_data):
    """Process the raw JSON data and extract clean text for each paper."""
    print("Processing raw Federalist data...")
    
    papers = []
    
    for paper_data in raw_data:
        try:
            # Debug: Print the keys in the first paper to understand structure
            if len(papers) == 0:
                print(f"  Debug: Available fields in paper data: {list(paper_data.keys())}")
            
            # Try to find text content in various possible fields
            text = None
            for field in ['text', 'content', 'body', 'document', 'paper_text']:
                if field in paper_data:
                    text = paper_data[field]
                    break
            
            # If no direct text field, check if we need to reconstruct from tokens
            if not text and 'tokens' in paper_data:
                # Reconstruct text from tokens (space-separated)
                tokens = paper_data['tokens']
                if isinstance(tokens, list):
                    text = ' '.join(tokens)
                    print(f"  ℹ Reconstructed text from {len(tokens)} tokens")
                
            if not text:
                print(f"  ⚠ Warning: No text content found in paper data. Available fields: {list(paper_data.keys())}")
                continue
            
            # Get metadata
            number = paper_data.get('number', 0)
            title = paper_data.get('title', f"Federalist Paper {number}")
            authors = paper_data.get('authors', 'Unknown')
            
            # Clean the text content
            content = clean_html_content(text) if text else ""
            
            if not content:
                print(f"  ⚠ Warning: No content extracted from paper {number}")
                continue
            
            # Show a preview of the content for debugging
            preview = content[:200] + "..." if len(content) > 200 else content
            preview = preview.replace('\n', ' ').strip()
            
            paper = {
                "number": number,
                "title": title,
                "authors": authors,
                "text": content
            }
            
            # Preserve original tokens field if it exists
            if 'tokens' in paper_data:
                paper['tokens'] = paper_data['tokens']
            
            papers.append(paper)
            print(f"  ✓ Processed Paper {number}: {title} by {authors}")
            print(f"    Preview: {preview}")
            
        except Exception as e:
            print(f"  ✗ Failed to process paper: {e}")
    
    print(f"\n✓ Processed {len(papers)} papers successfully")
    return papers

def save_raw_data(papers, output_dir=DEFAULT_OUTPUT_DIR):
    """Save the papers to JSON file in the Text subdirectory."""
    # Create the organized directory structure
    text_dir = os.path.join(output_dir, TEXT_DIR_PATH)
    os.makedirs(text_dir, exist_ok=True)
    
    output_file = os.path.join(text_dir, FEDERALIST_PAPERS_JSON)
    
    # Sort papers by number
    papers = sorted(papers, key=lambda x: x['number'])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(papers)} papers to {output_file}")
    
    # Print summary
    authors = {}
    for paper in papers:
        author = paper['authors']
        authors[author] = authors.get(author, 0) + 1
    
    print("\nAuthor distribution:")
    for author, count in sorted(authors.items()):
        print(f"  {author}: {count} papers")
    
    # Check for missing papers
    expected_numbers = set(range(1, 86))  # Papers 1-85
    actual_numbers = {paper['number'] for paper in papers}
    missing = expected_numbers - actual_numbers
    
    if missing:
        print(f"\nWarning: Missing papers: {sorted(missing)}")
    else:
        print("\n✓ All 85 papers downloaded successfully!")

def save_raw_html_files(output_dir=DEFAULT_OUTPUT_DIR):
    """Save the original files to the Raw subdirectory for archival."""
    raw_dir = os.path.join(output_dir, RAW_DIR_PATH)
    os.makedirs(raw_dir, exist_ok=True)
    
    files = get_federalist_file_list()
    if not files:
        print("No files to save as raw HTML")
        return
    
    print(f"Saving original files to {raw_dir}...")
    for file_info in files:
        filename = file_info['name']
        download_url = file_info['download_url']
        
        try:
            # Save original content
            with urllib.request.urlopen(download_url) as response:
                content = response.read()
            
            # Save with original filename
            file_path = os.path.join(raw_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            print(f"  ✓ Saved {filename}")
            
        except Exception as e:
            print(f"  ✗ Failed to save {filename}: {e}")

def main():
    """Main function."""
    print("Downloading and organizing Federalist Papers from nlp-course data repository...")
    
    # Step 1: Download the existing JSON file
    print("\n=== Step 1: Downloading federalist_data_raw2.json ===")
    raw_data = download_federalist_raw_json()
    
    if not raw_data:
        print("\n✗ Failed to download federalist data")
        return
    
    # Step 2: Process and extract clean text
    print("\n=== Step 2: Processing and extracting clean text ===")
    papers = process_raw_data(raw_data)
    
    if papers:
        save_raw_data(papers)
        print("\n✓ Federalist Papers data organized!")
        print(f"  - Clean text data in: {DEFAULT_OUTPUT_DIR}/{TEXT_DIR_PATH}/")
        print(f"  - Models will go in: {DEFAULT_OUTPUT_DIR}/{MODELS_DIR_PATH}/ (after training)")
        print("\nYou can now run: python build_federalist_data.py")
    else:
        print("\n✗ Failed to process papers")

if __name__ == "__main__":
    main()