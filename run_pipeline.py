#!/usr/bin/env python3
"""
KG Gap Discovery Engine - Main Pipeline Runner

Usage:
    python run_pipeline.py --stage collect        # Stage 1a: Collect papers
    python run_pipeline.py --stage filter         # Stage 1b: Filter corpus
    python run_pipeline.py --stage extract        # Stage 2: Extract triples
    python run_pipeline.py --stage build          # Stage 3: Build knowledge graph
    python run_pipeline.py --stage detect         # Stage 4: Detect gaps
    python run_pipeline.py --stage score          # Stage 5: Score and rank gaps
    python run_pipeline.py --stage visualise      # Generate visualisations
    python run_pipeline.py --stage all            # Run full pipeline
"""

import argparse
import yaml
import sys
import time
from pathlib import Path


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_stage(stage_name, config):
    """Run a specific pipeline stage."""
    
    print(f"\n{'='*60}")
    print(f"  Stage: {stage_name.upper()}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    if stage_name == "collect":
        from src.collect import collect_papers
        collect_papers(config)
    
    elif stage_name == "filter":
        from src.filter import filter_corpus
        filter_corpus(config)
    
    elif stage_name == "extract":
        from src.extract_triples import extract_all_triples
        extract_all_triples(config)
    
    elif stage_name == "build":
        from src.build_graph import build_knowledge_graph
        build_knowledge_graph(config)
    
    elif stage_name == "detect":
        from src.detect_gaps import detect_all_gaps
        detect_all_gaps(config)
    
    elif stage_name == "score":
        from src.score_gaps import score_and_rank_gaps
        score_and_rank_gaps(config)
    
    elif stage_name == "visualise":
        from src.visualise import generate_visualisations
        generate_visualisations(config)
    
    else:
        print(f"Unknown stage: {stage_name}")
        sys.exit(1)
    
    elapsed = time.time() - start_time
    print(f"\n  Completed '{stage_name}' in {elapsed:.1f} seconds.\n")


def main():
    parser = argparse.ArgumentParser(
        description="KG Gap Discovery Engine - Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages (run in order):
  collect    - Fetch papers from Semantic Scholar API
  filter     - Filter and clean the corpus
  extract    - Extract knowledge triples using LLM
  build      - Construct temporal knowledge graph
  detect     - Run gap detection algorithms
  score      - Score and rank detected gaps
  visualise  - Generate graph and gap visualisations
  all        - Run the complete pipeline
        """
    )
    parser.add_argument(
        "--stage", 
        required=True,
        choices=["collect", "filter", "extract", "build", "detect", "score", "visualise", "all"],
        help="Pipeline stage to run"
    )
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print(f"Make sure you're running from the project root directory.")
        sys.exit(1)
    
    config = load_config(args.config)
    
    print(f"\n  KG Gap Discovery Engine v{config['project']['version']}")
    print(f"  Domain: {config['project']['domain']}")
    print(f"  Config: {args.config}")
    
    if args.stage == "all":
        stages = ["collect", "filter", "extract", "build", "detect", "score", "visualise"]
        total_start = time.time()
        for stage in stages:
            run_stage(stage, config)
        total_elapsed = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"  FULL PIPELINE COMPLETED in {total_elapsed:.1f} seconds")
        print(f"{'='*60}\n")
    else:
        run_stage(args.stage, config)


if __name__ == "__main__":
    main()
