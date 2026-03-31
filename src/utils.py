"""
Shared utilities for the KG Gap Discovery pipeline.
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime


# --- Logging Setup ---

def get_logger(name, level=logging.INFO):
    """Create a configured logger."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


# --- File I/O ---

def save_json(data, filepath):
    """Save data as JSON with pretty formatting."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records, filepath):
    """Save records as JSON Lines (one JSON object per line)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(filepath):
    """Load records from JSON Lines file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --- Path Helpers ---

def get_project_root():
    """Get the project root directory."""
    # Assumes this file is in src/
    return Path(__file__).parent.parent


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


# --- Text Helpers ---

def clean_text(text):
    """Basic text cleaning for abstracts and paper content."""
    if not text:
        return ""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove common artifacts
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    return text.strip()


def chunk_text(text, chunk_size=1500, overlap=200):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


# --- Progress Helpers ---

def print_stage_header(stage_name, description=""):
    """Print a formatted stage header."""
    print(f"\n{'─'*50}")
    print(f"  {stage_name}")
    if description:
        print(f"  {description}")
    print(f"{'─'*50}\n")


def print_stats(stats_dict):
    """Print a formatted statistics summary."""
    max_key_len = max(len(k) for k in stats_dict.keys())
    for key, value in stats_dict.items():
        print(f"  {key:<{max_key_len + 2}} {value}")
    print()
