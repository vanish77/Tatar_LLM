# -*- coding: utf-8 -*-
"""
Script for downloading and preparing Tatar corpus
Downloads data from Leipzig Wortschatz and prepares it for training
"""

import os
import re
import requests
from pathlib import Path
from tqdm import tqdm
import tarfile
import gzip


def download_file(url, filename):
    """Download file with progress bar and validation"""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    
    # Check HTTP status
    response.raise_for_status()
    
    # Check content type
    content_type = response.headers.get('content-type', '')
    if 'html' in content_type.lower() or 'text' in content_type.lower():
        # Read first bytes to check if it's HTML
        first_bytes = response.content[:100]
        if first_bytes.startswith(b'<!') or first_bytes.startswith(b'<html'):
            raise ValueError(f"Downloaded file is HTML (likely 404 error), not an archive")
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
    
    # Validate that downloaded file is a valid tar.gz
    try:
        with tarfile.open(filename, 'r:gz') as tar:
            # Just check if we can read the archive
            tar.getmembers()
    except (tarfile.ReadError, gzip.BadGzipFile) as e:
        # Remove invalid file
        os.remove(filename)
        raise ValueError(f"Downloaded file is not a valid tar.gz archive: {e}")


def download_tatar_corpus():
    """Download Tatar corpus from Leipzig Wortschatz"""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # List of available Tatar language corpora
    # Updated with correct filenames from https://wortschatz.uni-leipzig.de/en/download/tat
    # Using only tat_mixed_2015_1M (192 MB, 1M sentences) for training
    # Note: If file already exists, it will be used; otherwise script will try to download
    corpora = [
        "tat_mixed_2015_1M.tar.gz",           # Mixed corpus - 1M sentences (~192 MB)
    ]
    
    # Try multiple possible base URLs
    base_urls = [
        "https://downloads.wortschatz-leipzig.de/corpora/",
        "https://wortschatz.uni-leipzig.de/download/",
    ]
    
    downloaded_files = []
    
    for corpus in corpora:
        file_path = data_dir / corpus
        if file_path.exists():
            # Validate existing file
            try:
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.getmembers()
                print(f"File {corpus} already exists and is valid, skipping download")
                downloaded_files.append(file_path)
                continue
            except (tarfile.ReadError, gzip.BadGzipFile, FileNotFoundError):
                print(f"File {corpus} exists but is corrupted, removing and re-downloading...")
                file_path.unlink(missing_ok=True)
            
        # Try downloading from different URLs
        downloaded = False
        for base_url in base_urls:
            url = f"{base_url}{corpus}"
            try:
                download_file(url, str(file_path))
                downloaded_files.append(file_path)
                print(f"Successfully downloaded {corpus} from {base_url}")
                downloaded = True
                break
            except Exception as e:
                print(f"Failed to download {corpus} from {base_url}: {e}")
                # Remove file if it was partially downloaded
                if file_path.exists():
                    file_path.unlink(missing_ok=True)
                continue
        
        if not downloaded:
            print(f"Could not download {corpus} from any source")
            print("Will use minimal test dataset if all downloads fail")
    
    return downloaded_files


def extract_corpus(tar_path):
    """Extract texts from tar.gz archive"""
    print(f"Extracting {tar_path}...")
    
    # Validate file before extraction
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.getmembers()
    except (tarfile.ReadError, gzip.BadGzipFile) as e:
        raise ValueError(f"File {tar_path} is not a valid tar.gz archive: {e}")
    
    extract_dir = tar_path.parent / tar_path.stem.replace('.tar', '')
    extract_dir.mkdir(exist_ok=True)
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    
    return extract_dir


def clean_text(text):
    """Clean text from extra characters"""
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text.strip()


def process_sentences_file(file_path):
    """Extract sentences from Leipzig format file"""
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Leipzig format: number\ttext
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    sentence = clean_text(parts[1])
                    if sentence and len(sentence) > 10:  # Filter too short sentences
                        sentences.append(sentence)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return sentences


def prepare_training_data():
    """Prepare data for training"""
    print("Preparing data for training...")
    
    # Download corpora
    downloaded_files = download_tatar_corpus()
    
    if not downloaded_files:
        print("Failed to download corpora. Creating minimal test dataset...")
        # Create minimal test dataset
        create_minimal_dataset()
        return
    
    all_sentences = []
    
    # Process each corpus
    for tar_path in downloaded_files:
        if not tar_path.exists():
            print(f"Skipping {tar_path} - file does not exist")
            continue
        
        # Check if already extracted
        extract_dir = tar_path.parent / tar_path.stem.replace('.tar', '')
        sentences_file = tar_path.parent / f"{tar_path.stem.replace('.tar', '')}-sentences.txt"
        
        # If sentences file already exists (extracted), use it directly
        if sentences_file.exists():
            print(f"Using already extracted file: {sentences_file.name}")
            sentences = process_sentences_file(sentences_file)
            all_sentences.extend(sentences)
            print(f"Extracted {len(sentences)} sentences from {sentences_file.name}")
            continue
        
        # Otherwise, extract from archive
        try:
            extract_dir = extract_corpus(tar_path)
        except (tarfile.ReadError, gzip.BadGzipFile, ValueError) as e:
            print(f"Error extracting {tar_path}: {e}")
            print(f"Removing corrupted file {tar_path}")
            tar_path.unlink(missing_ok=True)
            continue
        
        # Find sentences file
        for file_path in extract_dir.rglob("*-sentences.txt"):
            sentences = process_sentences_file(file_path)
            all_sentences.extend(sentences)
            print(f"Extracted {len(sentences)} sentences from {file_path.name}")
    
    # Save all sentences
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split into train/val
    total = len(all_sentences)
    train_size = int(total * 0.95)
    
    train_sentences = all_sentences[:train_size]
    val_sentences = all_sentences[train_size:]
    
    # Save
    with open(output_dir / "train.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_sentences))
    
    with open(output_dir / "val.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_sentences))
    
    # Statistics
    train_size_mb = os.path.getsize(output_dir / "train.txt") / (1024 * 1024)
    val_size_mb = os.path.getsize(output_dir / "val.txt") / (1024 * 1024)
    
    print("\n" + "="*50)
    print("Data preparation completed!")
    print(f"Total sentences: {total:,}")
    print(f"Train: {len(train_sentences):,} sentences ({train_size_mb:.2f} MB)")
    print(f"Val: {len(val_sentences):,} sentences ({val_size_mb:.2f} MB)")
    print(f"Files saved to: {output_dir}")
    print("="*50)


def create_minimal_dataset():
    """Create minimal test dataset if download failed"""
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample Tatar sentences for testing
    sample_sentences = [
        "Tatar tele - turki tellar turkemen kera.",
        "Kazan shehere Tatarstan Respublikasining bashkalasy.",
        "Min tatarcha oyranem.",
        "Bu kitap bik kyzykly.",
        "Haller nichek?",
    ] * 100  # Repeat for minimal size
    
    with open(output_dir / "train.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(sample_sentences))
    
    with open(output_dir / "val.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(sample_sentences[:20]))
    
    print("Created minimal test dataset")


if __name__ == "__main__":
    prepare_training_data()

