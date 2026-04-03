"""
Stage 4: Topological Gap Detection

Implements three gap detection algorithms on the knowledge graph:
1. Missing Link Prediction (TransE via PyKEEN)
2. Orphan Cluster Detection (Louvain community detection)
3. Temporal Decay Analysis

Usage:
    python run_pipeline.py --stage detect
"""

import pickle
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import community as community_louvain
from src.utils import get_logger, save_json, load_json, ensure_dir

logger = get_logger("detect_gaps")


# ============================================================
# GAP TYPE 1: Missing Link Prediction
# ============================================================

def detect_missing_links(G, config):
    """
    Use TransE graph embeddings to predict missing links.
    Entity pairs with high predicted scores but no existing edge
    are flagged as candidate gaps.
    """
    logger.info("--- Missing Link Prediction (TransE) ---")
    
    transE_config = config["gap_detection"]["transE"]
    top_k = transE_config["top_k_predictions"]
    
    try:
        from pykeen.pipeline import pipeline as pykeen_pipeline
        from pykeen.triples import TriplesFactory
    except ImportError:
        logger.error("PyKEEN not installed. Run: pip install pykeen")
        return []
    
    # Convert NetworkX graph to PyKEEN triples
    triples_list = []
    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "RELATED")
        triples_list.append([str(u), str(relation), str(v)])
    
    if len(triples_list) < 10:
        logger.warning(f"Too few triples ({len(triples_list)}) for meaningful link prediction")
        return []
    
    triples_array = np.array(triples_list)
    logger.info(f"  Training TransE on {len(triples_array)} triples...")
    
    # Create triples factory
    tf = TriplesFactory.from_labeled_triples(triples_array)
    
    # Train TransE model
    result = pykeen_pipeline(
        training=tf,
        model="TransE",
        model_kwargs={
            "embedding_dim": transE_config["embedding_dim"],
        },
        training_kwargs={
            "num_epochs": transE_config["num_epochs"],
            "use_tqdm": True,
        },
        optimizer_kwargs={
            "lr": transE_config["learning_rate"],
        },
        random_seed=42,
    )
    
    model = result.model
    logger.info(f"  TransE training complete. Loss: {result.losses[-1]:.4f}")
    
    # Predict missing links
    logger.info(f"  Predicting top {top_k} missing links...")
    
    # Get all existing edges as a set for fast lookup
    existing_edges = set()
    for u, v, data in G.edges(data=True):
        for rel in set(d.get("relation", "RELATED") for _, _, d in G.edges(u, v, data=True)):
            existing_edges.add((str(u), str(rel), str(v)))
    
    # Score all possible triples and find the best missing ones
    all_nodes = list(G.nodes())
    all_relations = list(set(d.get("relation", "RELATED") for _, _, d in G.edges(data=True)))
    
    # For efficiency, sample candidate pairs rather than all N^2
    candidates = []
    
    # Focus on node pairs that share a neighbor but aren't directly connected
    for node in all_nodes:
        neighbors = set(G.successors(node)) | set(G.predecessors(node))
        for neighbor in neighbors:
            second_hop = set(G.successors(neighbor)) | set(G.predecessors(neighbor))
            for target in second_hop:
                if target != node and not G.has_edge(node, target):
                    candidates.append((node, target))
    
    # Deduplicate candidates
    candidates = list(set(candidates))
    
    if not candidates:
        logger.warning("  No candidate missing links found")
        return []
    
    logger.info(f"  Evaluating {len(candidates)} candidate pairs...")
    
    # Score candidates using the trained model
    scored_gaps = []
    
    for head, tail in candidates[:min(len(candidates), 5000)]:  # Cap for performance
        for rel in all_relations:
            triple_key = (str(head), str(rel), str(tail))
            if triple_key in existing_edges:
                continue
            
            try:
                # Get score from model
                h_id = tf.entity_to_id.get(str(head))
                r_id = tf.relation_to_id.get(str(rel))
                t_id = tf.entity_to_id.get(str(tail))
                
                if h_id is None or r_id is None or t_id is None:
                    continue
                
                import torch
                h_tensor = torch.tensor([[h_id]])
                r_tensor = torch.tensor([[r_id]])
                t_tensor = torch.tensor([[t_id]])
                
                score = model.score_hrt(
                    torch.cat([h_tensor, r_tensor, t_tensor], dim=1)
                ).item()
                
                scored_gaps.append({
                    "type": "missing_link",
                    "head": str(head),
                    "relation": str(rel),
                    "tail": str(tail),
                    "prediction_score": float(score),
                    "description": f"Predicted connection: '{head}' --[{rel}]--> '{tail}' is likely but missing from the literature.",
                })
            except Exception:
                continue
    
    # Sort by score (higher = more likely missing link)
    scored_gaps.sort(key=lambda x: x["prediction_score"], reverse=True)
    top_gaps = scored_gaps[:top_k]
    
    logger.info(f"  Found {len(top_gaps)} missing link gaps")
    return top_gaps


# ============================================================
# GAP TYPE 2: Orphan Cluster Detection
# ============================================================

def detect_orphan_clusters(G, config):
    """
    Use Louvain community detection to find weakly connected
    subgraphs (orphan clusters) representing under-explored areas.
    """
    logger.info("--- Orphan Cluster Detection (Louvain) ---")
    
    orphan_config = config["gap_detection"]["orphan"]
    min_ratio = orphan_config["min_cluster_ratio"]
    max_inter_ratio = orphan_config["max_inter_cluster_edge_ratio"]
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Remove self-loops and isolates
    G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))
    isolates = list(nx.isolates(G_undirected))
    G_undirected.remove_nodes_from(isolates)
    
    if G_undirected.number_of_nodes() < 5:
        logger.warning("  Graph too small for community detection")
        return []
    
    # Run Louvain community detection
    # python-louvain needs a simple Graph (no multi-edges)
    G_simple = nx.Graph(G_undirected)
    partition = community_louvain.best_partition(G_simple)
    
    # Group nodes by community
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    total_nodes = G_simple.number_of_nodes()
    logger.info(f"  Detected {len(communities)} communities")
    
    # Analyse each community
    orphan_gaps = []
    
    for comm_id, members in communities.items():
        comm_size = len(members)
        size_ratio = comm_size / total_nodes
        
        # Count inter-community edges
        inter_edges = 0
        intra_edges = 0
        comm_set = set(members)
        
        for node in members:
            for neighbor in G_simple.neighbors(node):
                if neighbor in comm_set:
                    intra_edges += 1
                else:
                    inter_edges += 1
        
        intra_edges //= 2  # Undirected, counted twice
        total_comm_edges = inter_edges + intra_edges
        inter_ratio = inter_edges / max(total_comm_edges, 1)
        
        # Flag as orphan if small AND isolated
        if size_ratio < min_ratio or inter_ratio < max_inter_ratio:
            # Get the main concepts in this cluster
            node_types = defaultdict(list)
            for node in members:
                ntype = G.nodes[node].get("type", "UNKNOWN") if G.has_node(node) else "UNKNOWN"
                node_types[ntype].append(node)
            
            orphan_gaps.append({
                "type": "orphan_cluster",
                "community_id": comm_id,
                "size": comm_size,
                "size_ratio": round(size_ratio, 4),
                "inter_edge_ratio": round(inter_ratio, 4),
                "intra_edges": intra_edges,
                "inter_edges": inter_edges,
                "members": members,
                "key_concepts": members[:10],  # Top 10 for display
                "description": f"Isolated research cluster with {comm_size} concepts and only {inter_ratio:.1%} connections to the broader literature. Key concepts: {', '.join(members[:5])}.",
            })
    
    # Sort by isolation (lowest inter_ratio = most isolated)
    orphan_gaps.sort(key=lambda x: x["inter_edge_ratio"])
    
    logger.info(f"  Found {len(orphan_gaps)} orphan clusters")
    return orphan_gaps


# ============================================================
# GAP TYPE 3: Temporal Decay Analysis
# ============================================================

def detect_temporal_decay(G, config):
    """
    Identify concepts with declining research activity over time.
    Concepts with peak activity 2-3 years ago but declining
    recent edge formation are flagged.
    """
    logger.info("--- Temporal Decay Analysis ---")
    
    temp_config = config["gap_detection"]["temporal"]
    decay_threshold = temp_config["decay_threshold"]
    lookback = temp_config["lookback_years"]
    
    # Get year range from edges
    edge_years = []
    for _, _, data in G.edges(data=True):
        year = data.get("year")
        if year and isinstance(year, (int, float)):
            edge_years.append(int(year))
    
    if not edge_years:
        logger.warning("  No temporal data on edges")
        return []
    
    min_year = min(edge_years)
    max_year = max(edge_years)
    logger.info(f"  Temporal range: {min_year} - {max_year}")
    
    # Build per-node temporal profiles
    node_year_activity = defaultdict(lambda: defaultdict(int))
    
    for u, v, data in G.edges(data=True):
        year = data.get("year")
        if year and isinstance(year, (int, float)):
            year = int(year)
            node_year_activity[u][year] += 1
            node_year_activity[v][year] += 1
    
    # Analyse decay for each node
    decay_gaps = []
    recent_years = list(range(max_year - lookback + 1, max_year + 1))
    earlier_years = list(range(max_year - 2 * lookback + 1, max_year - lookback + 1))
    
    for node, year_counts in node_year_activity.items():
        recent_activity = sum(year_counts.get(y, 0) for y in recent_years)
        earlier_activity = sum(year_counts.get(y, 0) for y in earlier_years)
        
        # Skip nodes with very little activity overall
        total = sum(year_counts.values())
        if total < 3:
            continue
        
        # Calculate decay rate
        if earlier_activity > 0:
            decay_rate = 1.0 - (recent_activity / earlier_activity)
        elif recent_activity == 0:
            decay_rate = 1.0
        else:
            decay_rate = 0.0  # Growing, not decaying
        
        # Find peak year
        peak_year = max(year_counts, key=year_counts.get)
        peak_count = year_counts[peak_year]
        
        if decay_rate >= decay_threshold:
            # Build temporal profile for this node
            profile = {y: year_counts.get(y, 0) for y in range(min_year, max_year + 1)}
            
            decay_gaps.append({
                "type": "temporal_decay",
                "concept": node,
                "concept_type": G.nodes[node].get("type", "UNKNOWN") if G.has_node(node) else "UNKNOWN",
                "decay_rate": round(decay_rate, 4),
                "peak_year": peak_year,
                "peak_activity": peak_count,
                "recent_activity": recent_activity,
                "earlier_activity": earlier_activity,
                "total_activity": total,
                "temporal_profile": profile,
                "description": f"'{node}' peaked in {peak_year} with {peak_count} connections but has declined by {decay_rate:.0%} recently ({earlier_activity} -> {recent_activity} edges). This may indicate a stalled research thread worth revisiting.",
            })
    
    # Sort by decay rate (highest decay = most stalled)
    decay_gaps.sort(key=lambda x: x["decay_rate"], reverse=True)
    
    logger.info(f"  Found {len(decay_gaps)} decaying concepts")
    return decay_gaps


# ============================================================
# MAIN
# ============================================================

def detect_all_gaps(config):
    """
    Run all three gap detection algorithms and save results.
    """
    graph_dir = Path(config["paths"]["graph"])
    output_dir = ensure_dir(config["paths"]["outputs"])
    
    # Load knowledge graph
    pkl_path = graph_dir / "knowledge_graph.pkl"
    if not pkl_path.exists():
        logger.error(f"Knowledge graph not found: {pkl_path}")
        logger.error("Run 'python run_pipeline.py --stage build' first.")
        return
    
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    
    logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    all_gaps = {
        "missing_links": [],
        "orphan_clusters": [],
        "temporal_decay": [],
    }
    
    # --- Run detection algorithms ---
    
    # 1. Missing Link Prediction
    try:
        missing_links = detect_missing_links(G, config)
        all_gaps["missing_links"] = missing_links
    except Exception as e:
        logger.error(f"Missing link detection failed: {e}")
        logger.info("Continuing with other methods...")
    
    # 2. Orphan Cluster Detection
    try:
        orphan_clusters = detect_orphan_clusters(G, config)
        all_gaps["orphan_clusters"] = orphan_clusters
    except Exception as e:
        logger.error(f"Orphan cluster detection failed: {e}")
    
    # 3. Temporal Decay Analysis
    try:
        temporal_decay = detect_temporal_decay(G, config)
        all_gaps["temporal_decay"] = temporal_decay
    except Exception as e:
        logger.error(f"Temporal decay detection failed: {e}")
    
    # --- Save results ---
    save_json(all_gaps, output_dir / "detected_gaps_raw.json")
    
    # --- Print summary ---
    total = sum(len(v) for v in all_gaps.values())
    
    logger.info(f"\n{'='*50}")
    logger.info(f"  GAP DETECTION COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"  Missing links:    {len(all_gaps['missing_links'])}")
    logger.info(f"  Orphan clusters:  {len(all_gaps['orphan_clusters'])}")
    logger.info(f"  Temporal decay:   {len(all_gaps['temporal_decay'])}")
    logger.info(f"  Total gaps:       {total}")
    logger.info(f"  Saved to: {output_dir / 'detected_gaps_raw.json'}")
    
    return all_gaps


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    detect_all_gaps(config)
