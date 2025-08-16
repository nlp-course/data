#!/usr/bin/env python3
"""
Federalist Papers Gutenberg Download Script

This script downloads the Project Gutenberg version of the Federalist Papers
and saves it to the Raw directory for processing by prepare_federalist_data.py.

Usage:
    cd Federalist
    python Scripts/download_federalist_from_gutenberg.py

Output:
    Downloads the Gutenberg text file to Raw/federalist_papers_gutenberg.txt
"""

import os
import requests
from pathlib import Path

# Configuration
GUTENBERG_URL = "https://gutenberg.org/ebooks/1404.txt.utf-8"
OUTPUT_FILENAME = "federalist_papers_gutenberg.txt"

def download_gutenberg_federalist():
    """Download the Project Gutenberg Federalist Papers file."""
    print("Downloading Federalist Papers from Project Gutenberg...")
    print(f"URL: {GUTENBERG_URL}")
    
    try:
        # Create Raw directory if it doesn't exist
        raw_dir = Path("Raw")
        raw_dir.mkdir(exist_ok=True)
        
        # Download the file
        print("Downloading...")
        response = requests.get(GUTENBERG_URL, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save to Raw directory
        output_path = raw_dir / OUTPUT_FILENAME
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        print(f"\n✓ Downloaded successfully to: {output_path}")
        
        # Get file size
        file_size = output_path.stat().st_size
        print(f"File size: {file_size:,} bytes")
        
        return output_path
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading file: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None

def main():
    """Main execution function."""
    print("Federalist Papers Gutenberg Download")
    print("=" * 50)
    
    # Check if file already exists
    output_path = Path("Raw") / OUTPUT_FILENAME
    if output_path.exists():
        print(f"File already exists: {output_path}")
        response = input("Do you want to download again? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    # Download the file
    result = download_gutenberg_federalist()
    
    if result:
        print("\n✓ Download completed successfully!")
        print(f"File saved to: {result}")
        print("\nNext step: Run Scripts/prepare_federalist_data.py to process the Gutenberg file")
    else:
        print("\n✗ Download failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 