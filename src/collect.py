"""
Stage 1a: Data Collection from Semantic Scholar API

Fetches academic papers for the XAI domain using Semantic Scholar's
free API. No API key required (but rate-limited to ~100 req/5 min).

FIX 7 — 'Dynamic' justification:
  Added incremental update support so the graph can be updated as new
  papers are published without reprocessing the full corpus. Run with
  --incremental flag to collect only papers not already in the corpus.

Usage:
    python run_pipeline.py --stage collect
    python run_pipeline.py --stage collect --incremental
"""

import requests
import time
import json
from pathlib import Path
from tqdm import tqdm
from src.utils import get_logger, save_json, save_jsonl, load_jsonl, ensure_dir

logger = get_logger("collect")

# Semantic Scholar API endpoints
SEARCH_URL      = "https://api.semanticscholar.org/graph/v1/paper/search"
PAPER_URL       = "https://api.semanticscholar.org/graph/v1/paper"
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


# ── FIX 7: Incremental update helpers ─────────────────────────────────

def load_existing_paper_ids(raw_dir):
    """
    Load paper IDs already processed in a previous pipeline run.

    This enables the 'Dynamic' property of the KG: new papers published
    after the initial run can be incorporated without reprocessing the
    full corpus. The entity_name_map and existing graph are reused;
    only new triples are extracted and merged.

    Returns a set of paperId strings already in the corpus.
    """
    raw_dir = Path(raw_dir)
    all_jsonl = raw_dir / "all_papers_raw.jsonl"
    if not all_jsonl.exists():
        logger.info("  No existing corpus found — running full collection.")
        return set()

    existing = load_jsonl(all_jsonl)
    ids = {p.get("paperId") for p in existing if p.get("paperId")}
    logger.info(f"  Incremental mode: found {len(ids)} existing papers from previous run.")
    logger.info(f"  Only papers published after the last run will be collected.")
    return ids


def load_existing_papers(raw_dir):
    """
    Load the full list of already-collected papers.
    Used in incremental mode to merge new results with the existing corpus.
    """
    raw_dir = Path(raw_dir)
    all_jsonl = raw_dir / "all_papers_raw.jsonl"
    if not all_jsonl.exists():
        return []
    return load_jsonl(all_jsonl)


def save_incremental_manifest(raw_dir, run_info):
    """
    Save a timestamped manifest of each incremental run.
    Provides a longitudinal record of how the corpus has grown —
    supporting the 'temporal' aspect of the Dynamic KG.
    """
    raw_dir = Path(raw_dir)
    manifest_path = raw_dir / "incremental_runs.json"

    runs = []
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                runs = json.load(f)
        except Exception:
            runs = []

    runs.append(run_info)
    save_json(runs, manifest_path)
    logger.info(f"  Incremental run manifest updated: {manifest_path}")


# ── Core collection functions ─────────────────────────────────────────

def search_papers(query, year_range, api_key="", delay=3.5, max_results=500):
    """
    Search Semantic Scholar for papers matching the query.

    Uses the bulk search endpoint for larger result sets.
    Returns a list of paper dicts.
    """
    papers  = []
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query":  query,
        "fields": PAPER_FIELDS,
        "year":   f"{year_range[0]}-{year_range[1]}",
        "limit":  100,
    }

    logger.info(f"Searching: '{query}' ({year_range[0]}-{year_range[1]})")

    token = None
    page  = 0

    while True:
        if token:
            params["token"] = token

        try:
            response = requests.get(
                BULK_SEARCH_URL,
                params=params,
                headers=headers,
                timeout=30,
            )

            if response.status_code == 429:
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
        page  += 1
        total  = data.get("total", "?")
        logger.info(
            f"  Page {page}: fetched {len(batch)} papers "
            f"(total so far: {len(papers)}, available: {total})"
        )

        if len(papers) >= max_results:
            papers = papers[:max_results]
            break

        token = data.get("token")
        if not token:
            break

        time.sleep(delay)

    return papers


def deduplicate_papers(all_papers):
    """Remove duplicate papers based on paperId."""
    seen   = set()
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
    stats = {
        "no_abstract":    0,
        "short_abstract": 0,
        "no_title":       0,
        "no_year":        0,
        "passed":         0,
    }

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


# ── Main entry point ─────────────────────────────────────────────────

def collect_papers(config, incremental=False):
    """
    Main collection function. Runs all search queries, deduplicates,
    applies basic filters, and saves results.

    Parameters
    ----------
    config      : dict  — pipeline config from config.yaml
    incremental : bool  — if True, skip papers already in the corpus
                          (Fix 7: enables the 'Dynamic' KG property)
    """
    coll_config     = config["collection"]
    queries         = coll_config["queries"]
    year_range      = coll_config["year_range"]
    max_papers      = coll_config["max_papers"]
    delay           = coll_config["delay_between_requests"]
    api_key         = config["api_keys"].get("semantic_scholar", "")
    min_abstract_len= config["filtering"]["min_abstract_length"]

    raw_dir = ensure_dir(config["paths"]["raw_data"])

    # ── FIX 7: Load existing IDs if incremental mode ───────────────
    existing_ids    = set()
    existing_papers = []
    if incremental:
        logger.info("Running in INCREMENTAL mode (Fix 7 — Dynamic KG).")
        existing_ids    = load_existing_paper_ids(raw_dir)
        existing_papers = load_existing_papers(raw_dir)
        logger.info(
            f"  {len(existing_ids)} papers already in corpus. "
            f"Only new papers will be collected and processed."
        )
    # ──────────────────────────────────────────────────────────────

    logger.info(f"Starting collection for domain: {config['project']['domain']}")
    logger.info(f"Queries: {len(queries)}, Year range: {year_range}")

    all_papers      = []
    per_query_limit = max_papers // len(queries) + 100

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

        save_jsonl(papers, raw_dir / f"query_{i+1}_raw.jsonl")

        if i < len(queries) - 1:
            time.sleep(delay * 2)

    # ── Deduplicate across queries ─────────────────────────────────
    logger.info(f"\nTotal papers before deduplication: {len(all_papers)}")
    unique_papers = deduplicate_papers(all_papers)
    logger.info(f"Total papers after deduplication: {len(unique_papers)}")

    # ── FIX 7: Filter out already-processed papers in incremental mode
    if incremental and existing_ids:
        before_filter = len(unique_papers)
        unique_papers = [p for p in unique_papers if p.get("paperId") not in existing_ids]
        logger.info(
            f"  Incremental filter: {before_filter} → {len(unique_papers)} new papers "
            f"({before_filter - len(unique_papers)} already in corpus, skipped)"
        )
        if not unique_papers:
            logger.info("  No new papers found since the last run. Corpus is up to date.")
            return existing_papers
    # ──────────────────────────────────────────────────────────────

    # ── Basic quality filter ───────────────────────────────────────
    filtered_papers, filter_stats = basic_filter(unique_papers, min_abstract_len)

    logger.info(f"\nBasic filter results:")
    for key, val in filter_stats.items():
        logger.info(f"  {key}: {val}")

    # ── Sort by citation count ─────────────────────────────────────
    filtered_papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)

    # ── Cap to max_papers ─────────────────────────────────────────
    if len(filtered_papers) > max_papers:
        filtered_papers = filtered_papers[:max_papers]
        logger.info(f"Capped to {max_papers} papers")

    # ── FIX 7: In incremental mode, merge new papers with existing ─
    if incremental and existing_papers:
        merged = existing_papers + filtered_papers
        merged = deduplicate_papers(merged)
        logger.info(
            f"  Merged corpus: {len(existing_papers)} existing + "
            f"{len(filtered_papers)} new = {len(merged)} total papers"
        )
        filtered_papers = merged
    # ──────────────────────────────────────────────────────────────

    # ── Save ───────────────────────────────────────────────────────
    save_jsonl(filtered_papers, raw_dir / "all_papers_raw.jsonl")

    summary = []
    for p in filtered_papers:
        summary.append({
            "paperId":         p.get("paperId"),
            "title":           p.get("title"),
            "year":            p.get("year"),
            "citations":       p.get("citationCount", 0),
            "venue":           p.get("venue", ""),
            "abstract_length": len(p.get("abstract", "").split()),
        })
    save_json(summary, raw_dir / "papers_summary.json")

    # ── FIX 7: Save incremental run manifest ──────────────────────
    if incremental:
        import datetime
        run_info = {
            "timestamp":         datetime.datetime.now().isoformat(),
            "mode":              "incremental",
            "new_papers_added":  len(filtered_papers) - len(existing_papers),
            "total_corpus_size": len(filtered_papers),
            "year_range":        year_range,
            "queries":           queries,
        }
        save_incremental_manifest(raw_dir, run_info)
    # ──────────────────────────────────────────────────────────────

    # ── Print final stats ─────────────────────────────────────────
    years    = [p["year"] for p in filtered_papers if p.get("year")]
    year_dist = {}
    for y in years:
        year_dist[y] = year_dist.get(y, 0) + 1

    logger.info(f"\n{'='*50}")
    logger.info(f"  COLLECTION COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"  Total papers collected: {len(filtered_papers)}")
    if incremental:
        logger.info(f"  Mode: INCREMENTAL (Dynamic KG update)")
    logger.info(f"  Year distribution:")
    for y in sorted(year_dist.keys()):
        logger.info(f"    {y}: {year_dist[y]} papers")

    avg_citations = sum(p.get("citationCount", 0) for p in filtered_papers) / max(len(filtered_papers), 1)
    logger.info(f"  Average citations: {avg_citations:.1f}")
    logger.info(f"  Saved to: {raw_dir / 'all_papers_raw.jsonl'}")

    return filtered_papers


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Collect papers from Semantic Scholar")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Only collect papers not already in the corpus. "
            "Enables the Dynamic KG property: new papers are merged "
            "into the existing graph without full reprocessing. (Fix 7)"
        ),
    )
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    collect_papers(config, incremental=args.incremental)