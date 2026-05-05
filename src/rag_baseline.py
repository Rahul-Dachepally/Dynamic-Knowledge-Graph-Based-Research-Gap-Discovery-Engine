"""
RAG Baseline for Research Gap Identification

Implements two baselines to compare against the KG approach:

Method B: Mulla et al. (2026) RAG
  - Embed abstracts with Sentence-BERT
  - Retrieve top-3 similar papers per paper
  - Feed retrieved context to Llama for structured gap generation

Method C: Simple LLM baseline
  - Feed each abstract directly to Llama
  - Ask for gaps in free text

Usage:
    python run_rag_baseline.py
    or triggered from Streamlit app
"""

import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import get_logger, save_json, load_jsonl, ensure_dir

logger = get_logger("rag_baseline")

# ── Prompt: Mulla et al. exact format ──────────────────────
MULLA_PROMPT = """You are an expert academic research assistant. Your task is to analyze a research paper and provide a structured gap analysis.

Input:
- Research Topic: {topic}
- Paper Content: {text_chunk}
- Related Papers Context: {context}

Instructions:
1. Identify the RESEARCH_GAPS ADDRESSED: Specify the exact gaps or problems the paper targets.
2. Identify the RESEARCH_DIRECTION: Describe the general approach or methodology of the paper in 20-40 words.
3. Identify the SOLUTION_APPROACH: Describe in 60-100 words how the paper addresses the gaps, including methods, algorithms, frameworks, or experiments.
4. Identify REMAINING_GAPS: What gaps does this paper leave open or fail to address?

Provide output in this exact structured format:
RESEARCH_GAPS: [Detailed description, 80-120 words]
RESEARCH_DIRECTION: [20-40 words]
SOLUTION_APPROACH: [60-100 words]
REMAINING_GAPS: [40-80 words]

Rules:
- Do not add commentary outside these fields.
- Use complete sentences and concise wording.
- If information is missing, indicate "Not specified."
"""

# ── Prompt: Simple LLM baseline ────────────────────────────
SIMPLE_PROMPT = """You are a research assistant. Read the following paper abstract and identify research gaps.

Research Domain: {topic}
Paper Title: {title}
Abstract: {abstract}

List 3 specific research gaps this paper leaves open or that exist in this area.
Be concise and specific — each gap should be 1-2 sentences.

Format your response as JSON:
{{
  "gap_1": "...",
  "gap_2": "...",
  "gap_3": "..."
}}"""


# ── Helper: retry wrapper ───────────────────────────────────

def call_llm(client, model, system_msg, user_msg, max_tokens=600, max_retries=5):
    """Call LLM with automatic retry on rate limit."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e):
                wait = 15 * (attempt + 1)
                logger.warning(f"Rate limited. Waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            logger.warning(f"LLM call failed: {e}")
            return None
    return None


# ── Method B: Mulla et al. RAG ─────────────────────────────

def run_mulla_rag(papers, topic, client, model, output_path):
    """
    Replicate Mulla et al. (2026) RAG-based gap generation.
    Uses Sentence-BERT to retrieve top-3 similar papers as context.
    """
    logger.info("=== Method B: Mulla et al. RAG Baseline ===")

    # Build abstract corpus for retrieval
    abstracts = [p.get("abstract", "") for p in papers]
    titles    = [p.get("title", "")    for p in papers]

    # Embed all abstracts
    logger.info("Embedding abstracts with Sentence-BERT...")
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(abstracts, show_progress_bar=True, batch_size=32)

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    results = []

    for i, paper in enumerate(tqdm(papers, desc="Mulla RAG")):
        title    = titles[i]
        abstract = abstracts[i]

        if not abstract:
            continue

        # Retrieve top-3 similar papers (exclude self)
        sim_scores   = sim_matrix[i].copy()
        sim_scores[i] = -1  # exclude self
        top3_indices  = np.argsort(sim_scores)[-3:][::-1]

        context_parts = []
        for idx in top3_indices:
            context_parts.append(
                f"Related paper: {titles[idx]}\n"
                f"Abstract: {abstracts[idx][:300]}..."
            )
        context = "\n\n".join(context_parts)

        # Build prompt
        user_msg = MULLA_PROMPT.format(
            topic=topic,
            text_chunk=f"Title: {title}\nAbstract: {abstract}",
            context=context,
        )

        response = call_llm(
            client, model,
            system_msg="You are an expert academic research assistant. Follow the structured format exactly.",
            user_msg=user_msg,
            max_tokens=500,
        )

        if response:
            # Parse structured fields
            gap_result = {
                "paper_id":          paper.get("paperId", ""),
                "title":             title,
                "year":              paper.get("year"),
                "method":            "mulla_rag",
                "raw_response":      response,
                "research_gaps":     extract_field(response, "RESEARCH_GAPS"),
                "research_direction": extract_field(response, "RESEARCH_DIRECTION"),
                "solution_approach": extract_field(response, "SOLUTION_APPROACH"),
                "remaining_gaps":    extract_field(response, "REMAINING_GAPS"),
                "retrieved_context": [titles[idx] for idx in top3_indices],
            }
            results.append(gap_result)

        time.sleep(4)  # rate limit buffer

    save_json(results, output_path)
    logger.info(f"Mulla RAG: {len(results)} gap statements saved to {output_path}")
    return results


# ── Method C: Simple LLM ───────────────────────────────────

def run_simple_llm(papers, topic, client, model, output_path):
    """
    Simple LLM baseline — just ask Llama for gaps per abstract.
    No retrieval, no context, no structure enforcement.
    """
    logger.info("=== Method C: Simple LLM Baseline ===")

    results = []

    for paper in tqdm(papers, desc="Simple LLM"):
        title    = paper.get("title", "")
        abstract = paper.get("abstract", "")

        if not abstract:
            continue

        user_msg = SIMPLE_PROMPT.format(
            topic=topic,
            title=title,
            abstract=abstract[:800],  # truncate long abstracts
        )

        response = call_llm(
            client, model,
            system_msg="You are a research assistant. Respond only with valid JSON.",
            user_msg=user_msg,
            max_tokens=300,
        )

        gaps = {"gap_1": "", "gap_2": "", "gap_3": ""}
        if response:
            try:
                cleaned = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                gaps = json.loads(cleaned)
            except Exception:
                # If JSON fails, split by newline as fallback
                lines = [l.strip() for l in response.split("\n") if l.strip()]
                for j, line in enumerate(lines[:3], 1):
                    gaps[f"gap_{j}"] = line

        results.append({
            "paper_id": paper.get("paperId", ""),
            "title":    title,
            "year":     paper.get("year"),
            "method":   "simple_llm",
            "gap_1":    gaps.get("gap_1", ""),
            "gap_2":    gaps.get("gap_2", ""),
            "gap_3":    gaps.get("gap_3", ""),
            "all_gaps": [gaps.get(f"gap_{j}", "") for j in range(1, 4)],
        })

        time.sleep(4)  # rate limit buffer

    save_json(results, output_path)
    logger.info(f"Simple LLM: {len(results)} gap statements saved to {output_path}")
    return results


# ── Field extractor ─────────────────────────────────────────

def extract_field(text, field_name):
    """Extract a structured field from Mulla-format response."""
    try:
        start = text.find(f"{field_name}:")
        if start == -1:
            return ""
        start += len(f"{field_name}:")
        # Find next field or end of string
        next_field = -1
        for other in ["RESEARCH_GAPS:", "RESEARCH_DIRECTION:", "SOLUTION_APPROACH:", "REMAINING_GAPS:"]:
            if other == f"{field_name}:":
                continue
            pos = text.find(other, start)
            if pos != -1 and (next_field == -1 or pos < next_field):
                next_field = pos
        if next_field == -1:
            return text[start:].strip()
        return text[start:next_field].strip()
    except Exception:
        return ""


# ── Comparison metrics ──────────────────────────────────────

def compute_comparison_metrics(kg_gaps, mulla_gaps, simple_gaps):
    """
    Compute quantitative metrics to compare the three methods.
    """
    metrics = {}

    # --- KG metrics ---
    kg_descriptions = [g.get("description", "") for g in kg_gaps]
    metrics["kg"] = {
        "total_gaps":         len(kg_gaps),
        "missing_links":      sum(1 for g in kg_gaps if g["type"] == "missing_link"),
        "orphan_clusters":    sum(1 for g in kg_gaps if g["type"] == "orphan_cluster"),
        "temporal_decay":     sum(1 for g in kg_gaps if g["type"] == "temporal_decay"),
        "avg_score":          round(np.mean([g.get("composite_score", 0) for g in kg_gaps]), 4),
        "avg_description_len": round(np.mean([len(d.split()) for d in kg_descriptions if d]), 1),
        "unique_gaps":        len(set(kg_descriptions)),
        "has_evidence":       True,   # KG always has subgraph evidence
        "reproducible":       True,   # Graph structure is deterministic
    }

    # --- Mulla RAG metrics ---
    mulla_remaining = [g.get("remaining_gaps", "") for g in mulla_gaps if g.get("remaining_gaps")]
    metrics["mulla_rag"] = {
        "total_gaps":          len(mulla_gaps),
        "avg_gap_length":      round(np.mean([len(g.split()) for g in mulla_remaining if g]), 1) if mulla_remaining else 0,
        "unique_gaps":         len(set(mulla_remaining)),
        "papers_with_gaps":    sum(1 for g in mulla_gaps if g.get("remaining_gaps")),
        "has_evidence":        False,  # Free text, no traceable path
        "reproducible":        False,  # LLM is stochastic
        "uses_retrieval":      True,
    }

    # --- Simple LLM metrics ---
    all_simple_gaps = []
    for g in simple_gaps:
        all_simple_gaps.extend([g.get(f"gap_{j}", "") for j in range(1, 4)])
    all_simple_gaps = [g for g in all_simple_gaps if g]

    metrics["simple_llm"] = {
        "total_gaps":          len(all_simple_gaps),
        "unique_gaps":         len(set(all_simple_gaps)),
        "avg_gap_length":      round(np.mean([len(g.split()) for g in all_simple_gaps]), 1) if all_simple_gaps else 0,
        "papers_with_gaps":    len(simple_gaps),
        "has_evidence":        False,
        "reproducible":        False,
        "uses_retrieval":      False,
    }

    # --- Cross-method overlap (lexical) ---
    kg_words    = set(" ".join(kg_descriptions).lower().split())
    mulla_words = set(" ".join(mulla_remaining).lower().split())
    simple_words= set(" ".join(all_simple_gaps).lower().split())

    def jaccard(a, b):
        if not a or not b:
            return 0.0
        return round(len(a & b) / len(a | b), 4)

    metrics["overlap"] = {
        "kg_vs_mulla":  jaccard(kg_words, mulla_words),
        "kg_vs_simple": jaccard(kg_words, simple_words),
        "mulla_vs_simple": jaccard(mulla_words, simple_words),
    }

    return metrics


# ── Main entry point ────────────────────────────────────────

def run_rag_baseline(config):
    """
    Main function — runs both RAG baselines and computes comparison.
    Called from Streamlit or CLI.
    """
    proc_dir   = Path(config["paths"]["processed_data"])
    output_dir = ensure_dir(config["paths"]["outputs"])
    topic      = config["project"]["domain"]
    model      = config["extraction"]["model"]

    # Load filtered corpus
    corpus_path = proc_dir / "corpus_filtered.jsonl"
    if not corpus_path.exists():
        logger.error(f"Filtered corpus not found: {corpus_path}")
        return None

    papers = load_jsonl(corpus_path)
    logger.info(f"Loaded {len(papers)} papers for RAG baseline")

    # Init Groq client
    client = OpenAI(
        api_key=config["api_keys"]["groq"],
        base_url="https://api.groq.com/openai/v1",
    )

    # Run Method B: Mulla et al. RAG
    mulla_path  = output_dir / "rag_mulla_gaps.json"
    mulla_gaps  = run_mulla_rag(papers, topic, client, model, mulla_path)

    # Run Method C: Simple LLM
    simple_path = output_dir / "rag_simple_gaps.json"
    simple_gaps = run_simple_llm(papers, topic, client, model, simple_path)

    # Load KG gaps for comparison
    kg_path = output_dir / "gaps_ranked_top.json"
    kg_gaps = []
    if kg_path.exists():
        with open(kg_path) as f:
            kg_gaps = json.load(f)

    # Compute metrics
    metrics = compute_comparison_metrics(kg_gaps, mulla_gaps, simple_gaps)
    save_json(metrics, output_dir / "comparison_metrics.json")

    logger.info("\n" + "="*50)
    logger.info("  COMPARISON COMPLETE")
    logger.info("="*50)
    logger.info(f"  KG gaps:         {metrics['kg']['total_gaps']}")
    logger.info(f"  Mulla RAG gaps:  {metrics['mulla_rag']['total_gaps']}")
    logger.info(f"  Simple LLM gaps: {metrics['simple_llm']['total_gaps']}")
    logger.info(f"  KG vs Mulla overlap:  {metrics['overlap']['kg_vs_mulla']}")
    logger.info(f"  KG vs Simple overlap: {metrics['overlap']['kg_vs_simple']}")

    return {
        "mulla_gaps":  mulla_gaps,
        "simple_gaps": simple_gaps,
        "kg_gaps":     kg_gaps,
        "metrics":     metrics,
    }


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    run_rag_baseline(config)