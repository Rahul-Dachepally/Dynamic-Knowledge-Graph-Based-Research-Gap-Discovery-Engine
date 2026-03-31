#!/usr/bin/env python3
"""
Setup script for KG Gap Discovery project.
Run this once to create the full folder structure.

Usage:
    python setup_project.py
    # or
    python setup_project.py --path /your/preferred/location
"""

import os
import argparse

def create_project(base_path):
    """Create the full project directory structure."""
    
    dirs = [
        "data/raw",
        "data/processed",
        "data/triples",
        "data/graph",
        "src",
        "prompts",
        "notebooks",
        "outputs/figures",
    ]
    
    for d in dirs:
        path = os.path.join(base_path, d)
        os.makedirs(path, exist_ok=True)
        # Add .gitkeep to empty dirs so git tracks them
        gitkeep = os.path.join(path, ".gitkeep")
        if not os.path.exists(gitkeep):
            open(gitkeep, "w").close()
    
    # Create __init__.py for src/
    init_path = os.path.join(base_path, "src", "__init__.py")
    if not os.path.exists(init_path):
        open(init_path, "w").close()
    
    print(f"\n{'='*50}")
    print(f"  Project created at: {os.path.abspath(base_path)}")
    print(f"{'='*50}\n")
    print("Folder structure:")
    print_tree(base_path, prefix="")
    print(f"\nNext steps:")
    print(f"  1. cd {base_path}")
    print(f"  2. python -m venv venv")
    print(f"  3. source venv/bin/activate  (Linux/Mac)")
    print(f"     venv\\Scripts\\activate     (Windows)")
    print(f"  4. pip install -r requirements.txt")
    print(f"  5. Edit config.yaml with your API key")
    print(f"  6. python run_pipeline.py --stage collect")
    print()


def print_tree(path, prefix):
    """Print directory tree."""
    entries = sorted(os.listdir(path))
    entries = [e for e in entries if not e.startswith('.') and e != '__pycache__']
    
    for i, entry in enumerate(entries):
        full = os.path.join(path, entry)
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{entry}")
        if os.path.isdir(full):
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(full, next_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up KG Gap Discovery project")
    parser.add_argument("--path", default="kg-gap-discovery", help="Project root path")
    args = parser.parse_args()
    create_project(args.path)
