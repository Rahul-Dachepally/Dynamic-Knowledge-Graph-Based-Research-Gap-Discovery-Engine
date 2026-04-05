"""
Stage 1b: Corpus Filtering

Filters the raw collected papers down to the final corpus
using LLM-based relevance screening (mimicking PRISMA-style
title/abstract screening).

Usage:
    python run_pipeline.py --stage filter
"""

import json
import time
import os
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from src.utils import get_logger, save_json, save_jsonl, load_jsonl, ensure_dir

logger = get_logger("filter")
load_dotenv()  # Load environment variables from .env file

def load_prompt_template(prompts_dir):
    """Load the abstract filtering prompt template."""
    path = Path(prompts_dir) / "filter_abstract.txt"
    with open(path, "r") as f:
        return f.read()


def screen_paper(client, model, prompt_template, domain, title, abstract):
    """
    Use LLM to screen a single paper for relevance.
    Returns dict with relevant (bool), confidence (float), reason (str).
    Includes automatic retry with backoff for rate limiting.
    """
    prompt = prompt_template.replace("{domain}", str(domain))\
                            .replace("{title}", str(title))\
                            .replace("{abstract}", str(abstract))

    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research paper screening assistant. Respond only with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150,   # reduced from 200 to save tokens
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            return {
                "relevant": bool(result.get("relevant", False)),
                "confidence": float(result.get("confidence", 0.0)),
                "reason": str(result.get("reason", "")),
            }

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str:
                wait = 10 * (attempt + 1)   # 10s, 20s, 30s, 40s, 50s
                logger.warning(
                    f"Rate limited. Waiting {wait}s before retry "
                    f"{attempt + 1}/{max_retries}..."
                )
                time.sleep(wait)
                continue
            logger.warning(f"Screening failed for '{title[:50]}...': {e}")
            return {"relevant": False, "confidence": 0.0, "reason": f"Error: {e}"}

    return {"relevant": False, "confidence": 0.0, "reason": "Max retries exceeded"}


def filter_corpus(config):
    """
    Main filtering function.
    Loads raw papers, screens with LLM, filters to target corpus size.
    """
    raw_dir  = Path(config["paths"]["raw_data"])
    proc_dir = ensure_dir(config["paths"]["processed_data"])

    filter_config = config["filtering"]
    target_size   = filter_config["target_corpus_size"]
    model         = filter_config["relevance_model"]
    threshold     = filter_config["relevance_threshold"]
    domain        = config["project"]["domain"]

    # Load raw papers
    raw_path = raw_dir / "all_papers_raw.jsonl"
    if not raw_path.exists():
        logger.error(f"Raw data not found: {raw_path}")
        logger.error("Run 'python run_pipeline.py --stage collect' first.")
        return []

    papers = load_jsonl(raw_path)
    logger.info(f"Loaded {len(papers)} raw papers")

    # Load prompt template
    prompt_template = load_prompt_template(config["paths"]["prompts"])

    # Initialise client — Groq with OpenAI-compatible SDK
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    # --- Screen papers ---
    logger.info(f"Screening papers with {model} (threshold: {threshold})")
    logger.info(f"This may take a while for {len(papers)} papers...")

    screened        = []
    relevant_count  = 0

    for i, paper in enumerate(tqdm(papers, desc="Screening")):
        title    = paper.get("title", "")
        abstract = paper.get("abstract", "")

        if not abstract:
            continue

        result = screen_paper(client, model, prompt_template, domain, title, abstract)

        paper["screening"] = result
        screened.append(paper)

        if result["relevant"] and result["confidence"] >= threshold:
            relevant_count += 1

        # Save progress every 50 papers
        if (i + 1) % 50 == 0:
            save_jsonl(screened, proc_dir / "screening_progress.jsonl")
            logger.info(
                f"  Progress: {i + 1}/{len(papers)}, "
                f"relevant so far: {relevant_count}"
            )

        # Rate limiting — 3 seconds keeps us safely under 6k TPM on free tier
        time.sleep(3)

    # --- Filter to relevant papers ---
    relevant_papers = [
        p for p in screened
        if p["screening"]["relevant"] and p["screening"]["confidence"] >= threshold
    ]

    logger.info(f"\nScreening complete:")
    logger.info(f"  Total screened:            {len(screened)}")
    logger.info(f"  Relevant (above threshold): {len(relevant_papers)}")

    # --- Sort by confidence * citations to get best papers ---
    relevant_papers.sort(
        key=lambda p: (
            p["screening"]["confidence"]
            * (1 + p.get("citationCount", 0) ** 0.5)
        ),
        reverse=True,
    )

    # --- Cap to target corpus size ---
    if len(relevant_papers) > target_size:
        final_corpus = relevant_papers[:target_size]
        logger.info(f"  Capped to target size: {target_size}")
    else:
        final_corpus = relevant_papers
        logger.info(
            f"  Final corpus size: {len(final_corpus)} "
            f"(target was {target_size})"
        )

    # --- Save results ---
    save_jsonl(final_corpus, proc_dir / "corpus_filtered.jsonl")
    save_jsonl(screened,     proc_dir / "all_screened.jsonl")

    # Save human-readable summary
    summary = []
    for p in final_corpus:
        summary.append({
            "paperId":        p.get("paperId"),
            "title":          p.get("title"),
            "year":           p.get("year"),
            "citations":      p.get("citationCount", 0),
            "relevance_score": p["screening"]["confidence"],
            "reason":         p["screening"]["reason"],
        })
    save_json(summary, proc_dir / "corpus_summary.json")

    # --- Print stats ---
    years    = [p["year"] for p in final_corpus if p.get("year")]
    year_dist = {}
    for y in years:
        year_dist[y] = year_dist.get(y, 0) + 1

    logger.info(f"\n{'='*50}")
    logger.info(f"  FILTERING COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"  Final corpus: {len(final_corpus)} papers")
    logger.info(f"  Year distribution:")
    for y in sorted(year_dist.keys()):
        logger.info(f"    {y}: {year_dist[y]} papers")
    logger.info(f"  Saved to: {proc_dir / 'corpus_filtered.jsonl'}")

    return final_corpus


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    filter_corpus(config)