"""
Stage 1a: Data Collection from Semantic Scholar API

Fetches academic papers for the XAI domain using Semantic Scholar's
free API. No API key required (but rate-limited to ~100 req/5 min).

Usage:
    python run_pipeline.py --stage collect
"""

import requests
import time
import json
from pathlib import Path
from tqdm import tqdm
from src.utils import get_logger, save_json, save_jsonl, load_jsonl, ensure_dir

logger = get_logger("collect")

# Semantic Scholar API endpoints
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

# Fields to fetch for each paper
PAPER_FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "year",
    "citationCount",
    "referenceCount",
    "fieldsOfStudy",
    "publicationTypes",
    "journal",
    "venue",
    "authors",
    "externalIds",
    "url",
    "openAccessPdf",
    "publicationDate",
])


def search_papers(query, year_range, api_key="", delay=3.5, max_results=500):
    """
    Search Semantic Scholar for papers matching the query.
    
    Uses the bulk search endpoint for larger result sets.
    Returns a list of paper dicts.
    """
    papers = []
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    params = {
        "query": query,
        "fields": PAPER_FIELDS,
        "year": f"{year_range[0]}-{year_range[1]}",
        "limit": 100,  # Max per request
    }
    
    logger.info(f"Searching: '{query}' ({year_range[0]}-{year_range[1]})")
    
    token = None
    page = 0
    
    while True:
        if token:
            params["token"] = token
        
        try:
            response = requests.get(
                BULK_SEARCH_URL, 
                params=params, 
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 429:
                # Rate limited - wait and retry
                wait = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            
            response.raise_for_status()
            data = response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API error: {e}")
            time.sleep(delay * 2)
            continue
        
        batch = data.get("data", [])
        if not batch:
            break
        
        papers.extend(batch)
        page += 1
        total = data.get("total", "?")
        logger.info(f"  Page {page}: fetched {len(batch)} papers (total so far: {len(papers)}, available: {total})")
        
        if len(papers) >= max_results:
            papers = papers[:max_results]
            break
        
        # Check for next page
        token = data.get("token")
        if not token:
            break
        
        time.sleep(delay)
    
    return papers


def deduplicate_papers(all_papers):
    """Remove duplicate papers based on paperId."""
    seen = set()
    unique = []
    for paper in all_papers:
        pid = paper.get("paperId")
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(paper)
    return unique


def basic_filter(papers, min_abstract_length=100):
    """
    Apply basic quality filters:
    - Must have abstract
    - Abstract must meet minimum length
    - Must have a title
    - Must have a year
    """
    filtered = []
    stats = {"no_abstract": 0, "short_abstract": 0, "no_title": 0, "no_year": 0, "passed": 0}
    
    for paper in papers:
        if not paper.get("abstract"):
            stats["no_abstract"] += 1
            continue
        
        abstract_words = len(paper["abstract"].split())
        if abstract_words < min_abstract_length:
            stats["short_abstract"] += 1
            continue
        
        if not paper.get("title"):
            stats["no_title"] += 1
            continue
        
        if not paper.get("year"):
            stats["no_year"] += 1
            continue
        
        stats["passed"] += 1
        filtered.append(paper)
    
    return filtered, stats


def collect_papers(config):
    """
    Main collection function. Runs all search queries,
    deduplicates, applies basic filters, and saves results.
    """
    coll_config = config["collection"]
    queries = coll_config["queries"]
    year_range = coll_config["year_range"]
    max_papers = coll_config["max_papers"]
    delay = coll_config["delay_between_requests"]
    api_key = config["api_keys"].get("semantic_scholar", "")
    min_abstract_len = config["filtering"]["min_abstract_length"]
    
    raw_dir = ensure_dir(config["paths"]["raw_data"])
    
    # --- Step 1: Search across all queries ---
    logger.info(f"Starting collection for domain: {config['project']['domain']}")
    logger.info(f"Queries: {len(queries)}, Year range: {year_range}")
    
    all_papers = []
    per_query_limit = max_papers // len(queries) + 100  # Extra buffer per query
    
    for i, query in enumerate(queries):
        logger.info(f"\n--- Query {i+1}/{len(queries)} ---")
        papers = search_papers(
            query=query,
            year_range=year_range,
            api_key=api_key,
            delay=delay,
            max_results=per_query_limit,
        )
        logger.info(f"  Retrieved {len(papers)} papers for '{query}'")
        all_papers.extend(papers)
        
        # Save per-query results
        save_jsonl(papers, raw_dir / f"query_{i+1}_raw.jsonl")
        
        # Be nice to the API between queries
        if i < len(queries) - 1:
            time.sleep(delay * 2)
    
    # --- Step 2: Deduplicate ---
    logger.info(f"\nTotal papers before deduplication: {len(all_papers)}")
    unique_papers = deduplicate_papers(all_papers)
    logger.info(f"Total papers after deduplication: {len(unique_papers)}")
    
    # --- Step 3: Basic quality filter ---
    filtered_papers, filter_stats = basic_filter(unique_papers, min_abstract_len)
    
    logger.info(f"\nBasic filter results:")
    for key, val in filter_stats.items():
        logger.info(f"  {key}: {val}")
    
    # --- Step 4: Sort by citation count (most cited first) ---
    filtered_papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
    
    # --- Step 5: Cap at max_papers ---
    if len(filtered_papers) > max_papers:
        filtered_papers = filtered_papers[:max_papers]
        logger.info(f"Capped to {max_papers} papers")
    
    # --- Step 6: Save ---
    save_jsonl(filtered_papers, raw_dir / "all_papers_raw.jsonl")
    
    # Save a lightweight summary CSV-like JSON for quick inspection
    summary = []
    for p in filtered_papers:
        summary.append({
            "paperId": p.get("paperId"),
            "title": p.get("title"),
            "year": p.get("year"),
            "citations": p.get("citationCount", 0),
            "venue": p.get("venue", ""),
            "abstract_length": len(p.get("abstract", "").split()),
        })
    save_json(summary, raw_dir / "papers_summary.json")
    
    # --- Print final stats ---
    years = [p["year"] for p in filtered_papers if p.get("year")]
    year_dist = {}
    for y in years:
        year_dist[y] = year_dist.get(y, 0) + 1
    
    logger.info(f"\n{'='*50}")
    logger.info(f"  COLLECTION COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"  Total papers collected: {len(filtered_papers)}")
    logger.info(f"  Year distribution:")
    for y in sorted(year_dist.keys()):
        logger.info(f"    {y}: {year_dist[y]} papers")
    
    avg_citations = sum(p.get("citationCount", 0) for p in filtered_papers) / max(len(filtered_papers), 1)
    logger.info(f"  Average citations: {avg_citations:.1f}")
    logger.info(f"  Saved to: {raw_dir / 'all_papers_raw.jsonl'}")
    
    return filtered_papers


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    collect_papers(config)
