"""
Stage 5: Gap Confidence Scoring and Ranking

Combines metrics from all three detection methods into a composite
confidence score and produces the final ranked gap list.

Usage:
    python run_pipeline.py --stage score
"""

import pickle
import networkx as nx
import pandas as pd
from pathlib import Path
from src.utils import get_logger, save_json, load_json, ensure_dir

logger = get_logger("score_gaps")


def score_missing_link(gap, G):
    """Score a missing link gap using graph topology + prediction score."""
    head = gap.get("head", "")
    tail = gap.get("tail", "")
    pred_score = gap.get("prediction_score", 0.0)
    
    # Normalise prediction score to [0, 1]
    norm_pred = min(abs(pred_score) / 10.0, 1.0)
    
    # Get centrality of involved nodes
    head_cent = G.nodes[head].get("degree_centrality", 0.0) if G.has_node(head) else 0.0
    tail_cent = G.nodes[tail].get("degree_centrality", 0.0) if G.has_node(tail) else 0.0
    avg_centrality = (head_cent + tail_cent) / 2.0
    
    return {
        "prediction_confidence": norm_pred,
        "centrality": avg_centrality,
        "cluster_isolation": 0.0,  # Not applicable for missing links
        "temporal_decay": 0.0,     # Not applicable for missing links
    }


def score_orphan_cluster(gap, G):
    """Score an orphan cluster gap using isolation metrics."""
    inter_ratio = gap.get("inter_edge_ratio", 0.0)
    size_ratio = gap.get("size_ratio", 0.0)
    
    # More isolated = higher gap score
    isolation = 1.0 - inter_ratio
    
    # Get average centrality of cluster members
    members = gap.get("members", [])
    centralities = [
        G.nodes[m].get("degree_centrality", 0.0)
        for m in members if G.has_node(m)
    ]
    avg_centrality = sum(centralities) / max(len(centralities), 1)
    
    return {
        "prediction_confidence": 0.0,  # Not applicable
        "centrality": avg_centrality,
        "cluster_isolation": isolation,
        "temporal_decay": 0.0,  # Not applicable
    }


def score_temporal_decay(gap, G):
    """Score a temporal decay gap using decline metrics."""
    decay_rate = gap.get("decay_rate", 0.0)
    concept = gap.get("concept", "")
    
    centrality = G.nodes[concept].get("degree_centrality", 0.0) if G.has_node(concept) else 0.0
    
    return {
        "prediction_confidence": 0.0,  # Not applicable
        "centrality": centrality,
        "cluster_isolation": 0.0,  # Not applicable
        "temporal_decay": decay_rate,
    }


def compute_composite_score(metrics, weights):
    """Compute weighted composite score."""
    score = 0.0
    for key, weight in weights.items():
        score += weight * metrics.get(key, 0.0)
    return score


def score_and_rank_gaps(config):
    """
    Main scoring function. Loads raw gaps, computes composite scores,
    and produces the final ranked output.
    """
    output_dir = Path(config["paths"]["outputs"])
    graph_dir = Path(config["paths"]["graph"])
    
    scoring_config = config["scoring"]
    weights = scoring_config["weights"]
    top_k = scoring_config["top_k_gaps"]
    
    # Load knowledge graph
    pkl_path = graph_dir / "knowledge_graph.pkl"
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    
    # Load raw gaps
    gaps_path = output_dir / "detected_gaps_raw.json"
    if not gaps_path.exists():
        logger.error(f"Raw gaps not found: {gaps_path}")
        logger.error("Run 'python run_pipeline.py --stage detect' first.")
        return
    
    raw_gaps = load_json(gaps_path)
    
    # --- Score all gaps ---
    scored_gaps = []
    
    # Missing links
    for gap in raw_gaps.get("missing_links", []):
        metrics = score_missing_link(gap, G)
        composite = compute_composite_score(metrics, weights)
        scored_gaps.append({
            **gap,
            "metrics": metrics,
            "composite_score": round(composite, 4),
        })
    
    # Orphan clusters
    for gap in raw_gaps.get("orphan_clusters", []):
        metrics = score_orphan_cluster(gap, G)
        composite = compute_composite_score(metrics, weights)
        scored_gaps.append({
            **gap,
            "metrics": metrics,
            "composite_score": round(composite, 4),
        })
    
    # Temporal decay
    for gap in raw_gaps.get("temporal_decay", []):
        metrics = score_temporal_decay(gap, G)
        composite = compute_composite_score(metrics, weights)
        scored_gaps.append({
            **gap,
            "metrics": metrics,
            "composite_score": round(composite, 4),
        })
    
    # --- Rank by composite score ---
    scored_gaps.sort(key=lambda x: x["composite_score"], reverse=True)
    
    # Add rank
    for i, gap in enumerate(scored_gaps):
        gap["rank"] = i + 1
    
    # --- Take top K ---
    top_gaps = scored_gaps[:top_k]
    
    # --- Save results ---
    save_json(scored_gaps, output_dir / "gaps_scored_all.json")
    save_json(top_gaps, output_dir / "gaps_ranked_top.json")
    
    # Save a clean CSV for easy viewing
    csv_rows = []
    for gap in top_gaps:
        csv_rows.append({
            "rank": gap["rank"],
            "type": gap["type"],
            "score": gap["composite_score"],
            "description": gap.get("description", "")[:200],
        })
    
    df = pd.DataFrame(csv_rows)
    df.to_csv(output_dir / "gaps_ranked.csv", index=False)
    
    # --- Print results ---
    logger.info(f"\n{'='*50}")
    logger.info(f"  GAP SCORING COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"  Total scored: {len(scored_gaps)}")
    logger.info(f"  Top {top_k} gaps saved")
    logger.info(f"\n  Top 10 gaps:")
    
    for gap in top_gaps[:10]:
        logger.info(f"    #{gap['rank']} [{gap['type']}] (score: {gap['composite_score']:.3f})")
        logger.info(f"       {gap.get('description', '')[:100]}...")
    
    logger.info(f"\n  Saved to:")
    logger.info(f"    {output_dir / 'gaps_ranked.csv'}")
    logger.info(f"    {output_dir / 'gaps_ranked_top.json'}")
    
    return top_gaps


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    score_and_rank_gaps(config)
