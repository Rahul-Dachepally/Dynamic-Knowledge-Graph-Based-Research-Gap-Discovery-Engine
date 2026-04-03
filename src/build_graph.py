"""
Stage 3: Temporal Knowledge Graph Construction

Takes extracted triples, deduplicates entities using fuzzy matching
and semantic similarity, and builds a NetworkX graph.

Usage:
    python run_pipeline.py --stage build
"""

import pickle
import networkx as nx
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import get_logger, save_json, load_json, ensure_dir

logger = get_logger("build_graph")


def build_entity_index(triples):
    """
    Build an index of all unique entities from triples.
    Returns dict: entity_name -> {type, occurrences, papers}
    """
    entities = {}
    
    for t in triples:
        for role in ["subject", "object"]:
            name = t[role]["name"].strip()
            etype = t[role].get("type", "UNKNOWN")
            paper_id = t.get("source_paper_id", "")
            
            if name not in entities:
                entities[name] = {
                    "type": etype,
                    "occurrences": 0,
                    "papers": set(),
                    "original_name": name,
                }
            entities[name]["occurrences"] += 1
            if paper_id:
                entities[name]["papers"].add(paper_id)
    
    # Convert sets to lists for serialisation
    for e in entities.values():
        e["papers"] = list(e["papers"])
    
    return entities


def deduplicate_entities(entities, fuzzy_threshold=85, semantic_threshold=0.85, embedding_model_name="all-MiniLM-L6-v2"):
    """
    Deduplicate entities using two-pass approach:
    1. Fuzzy string matching
    2. Semantic similarity with Sentence-BERT
    
    Returns a mapping: original_name -> canonical_name
    """
    names = list(entities.keys())
    logger.info(f"Deduplicating {len(names)} entities...")
    
    # Mapping from original name to canonical (merged) name
    name_map = {n: n for n in names}
    
    # --- Pass 1: Fuzzy string matching ---
    logger.info("  Pass 1: Fuzzy string matching...")
    merged_fuzzy = 0
    
    # Sort by occurrence count (most frequent = canonical)
    sorted_names = sorted(names, key=lambda n: entities[n]["occurrences"], reverse=True)
    
    for i, name_a in enumerate(sorted_names):
        if name_map[name_a] != name_a:
            continue  # Already merged
        
        for name_b in sorted_names[i+1:]:
            if name_map[name_b] != name_b:
                continue  # Already merged
            
            score = fuzz.token_sort_ratio(name_a.lower(), name_b.lower())
            if score >= fuzzy_threshold:
                # Merge name_b into name_a (name_a has more occurrences)
                name_map[name_b] = name_a
                merged_fuzzy += 1
    
    logger.info(f"    Fuzzy merged: {merged_fuzzy} entities")
    
    # --- Pass 2: Semantic similarity ---
    logger.info("  Pass 2: Semantic similarity matching...")
    
    # Get remaining unique canonical names
    canonical_names = list(set(name_map.values()))
    
    if len(canonical_names) > 1:
        # Load Sentence-BERT model
        logger.info(f"    Loading embedding model: {embedding_model_name}")
        model = SentenceTransformer(embedding_model_name)
        
        # Encode all canonical names
        logger.info(f"    Encoding {len(canonical_names)} entity names...")
        embeddings = model.encode(canonical_names, show_progress_bar=False, batch_size=64)
        
        # Compute pairwise cosine similarity
        sim_matrix = cosine_similarity(embeddings)
        
        merged_semantic = 0
        # Sort canonical names by occurrence for consistent canonical selection
        canonical_occurrences = {
            n: entities.get(n, {}).get("occurrences", 0) for n in canonical_names
        }
        canonical_sorted = sorted(canonical_names, key=lambda n: canonical_occurrences.get(n, 0), reverse=True)
        canonical_idx = {n: i for i, n in enumerate(canonical_names)}
        
        semantic_map = {n: n for n in canonical_names}
        
        for name_a in canonical_sorted:
            if semantic_map[name_a] != name_a:
                continue
            
            idx_a = canonical_idx[name_a]
            
            for name_b in canonical_sorted:
                if name_b == name_a or semantic_map[name_b] != name_b:
                    continue
                
                idx_b = canonical_idx[name_b]
                
                if sim_matrix[idx_a][idx_b] >= semantic_threshold:
                    semantic_map[name_b] = name_a
                    merged_semantic += 1
        
        # Update the main name_map with semantic merges
        for orig_name, canon_name in name_map.items():
            if canon_name in semantic_map:
                name_map[orig_name] = semantic_map[canon_name]
        
        logger.info(f"    Semantic merged: {merged_semantic} entities")
    
    unique_final = len(set(name_map.values()))
    logger.info(f"  Final unique entities: {unique_final} (from {len(names)})")
    
    return name_map


def build_knowledge_graph(config):
    """
    Main graph construction function.
    Loads triples, deduplicates entities, builds NetworkX graph.
    """
    triples_dir = Path(config["paths"]["triples"])
    graph_dir = ensure_dir(config["paths"]["graph"])
    graph_config = config["graph"]
    
    # Load all triples
    triples_path = triples_dir / "all_triples.json"
    if not triples_path.exists():
        logger.error(f"Triples not found: {triples_path}")
        logger.error("Run 'python run_pipeline.py --stage extract' first.")
        return
    
    triples = load_json(triples_path)
    logger.info(f"Loaded {len(triples)} triples")
    
    # Filter low-confidence triples
    min_conf = graph_config["min_edge_confidence"]
    triples = [t for t in triples if t.get("confidence", 0) >= min_conf]
    logger.info(f"After confidence filter (>={min_conf}): {len(triples)} triples")
    
    # --- Step 1: Build entity index ---
    entities = build_entity_index(triples)
    logger.info(f"Raw unique entities: {len(entities)}")
    
    # --- Step 2: Deduplicate entities ---
    name_map = deduplicate_entities(
        entities,
        fuzzy_threshold=graph_config["fuzzy_match_threshold"],
        semantic_threshold=graph_config["semantic_similarity_threshold"],
        embedding_model_name=graph_config["embedding_model"],
    )
    
    # --- Step 3: Build NetworkX graph ---
    logger.info("Building knowledge graph...")
    
    G = nx.MultiDiGraph()
    
    edge_count = 0
    skipped = 0
    
    for t in triples:
        subj_raw = t["subject"]["name"].strip()
        obj_raw = t["object"]["name"].strip()
        
        # Apply deduplication mapping
        subj = name_map.get(subj_raw, subj_raw)
        obj = name_map.get(obj_raw, obj_raw)
        
        # Skip self-loops
        if subj == obj:
            skipped += 1
            continue
        
        relation = t.get("relation", "RELATED")
        confidence = t.get("confidence", 0.5)
        paper_id = t.get("source_paper_id", "")
        year = t.get("source_year", None)
        
        # Add nodes with attributes
        if not G.has_node(subj):
            G.add_node(subj, type=t["subject"].get("type", "UNKNOWN"), papers=set())
        G.nodes[subj]["papers"].add(paper_id)
        
        if not G.has_node(obj):
            G.add_node(obj, type=t["object"].get("type", "UNKNOWN"), papers=set())
        G.nodes[obj]["papers"].add(paper_id)
        
        # Add edge with metadata
        G.add_edge(
            subj, obj,
            relation=relation,
            confidence=confidence,
            source_paper=paper_id,
            year=year,
            evidence=t.get("evidence", ""),
        )
        edge_count += 1
    
    # Convert paper sets to lists for serialisation
    for node in G.nodes():
        G.nodes[node]["papers"] = list(G.nodes[node]["papers"])
    
    logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logger.info(f"Skipped self-loops: {skipped}")
    
    # --- Step 4: Compute basic graph metrics ---
    G_simple = nx.DiGraph(G)  # Collapse multi-edges for metrics
    
    # Degree centrality
    degree_cent = nx.degree_centrality(G_simple)
    for node, cent in degree_cent.items():
        G.nodes[node]["degree_centrality"] = cent
    
    # Betweenness centrality (on simple graph)
    if G_simple.number_of_nodes() > 1:
        betweenness = nx.betweenness_centrality(G_simple)
        for node, cent in betweenness.items():
            G.nodes[node]["betweenness_centrality"] = cent
    
    # --- Step 5: Save ---
    # Save as GraphML (widely compatible)
    graphml_path = graph_dir / "knowledge_graph.graphml"
    # GraphML can't handle sets/lists in attributes, so convert
    G_save = G.copy()
    for node in G_save.nodes():
        G_save.nodes[node]["papers"] = ",".join(G_save.nodes[node].get("papers", []))
    for u, v, k, data in G_save.edges(keys=True, data=True):
        for key, val in data.items():
            if val is None:
                data[key] = ""
    nx.write_graphml(G_save, graphml_path)
    
    # Save as pickle (preserves all Python types)
    pickle_path = graph_dir / "knowledge_graph.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(G, f)
    
    # Save entity mapping
    save_json(name_map, graph_dir / "entity_name_map.json")
    
    # --- Print stats ---
    node_types = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_types[data.get("type", "UNKNOWN")] += 1
    
    edge_relations = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_relations[data.get("relation", "UNKNOWN")] += 1
    
    years_on_edges = [d.get("year") for _, _, d in G.edges(data=True) if d.get("year")]
    
    logger.info(f"\n{'='*50}")
    logger.info(f"  KNOWLEDGE GRAPH BUILT")
    logger.info(f"{'='*50}")
    logger.info(f"  Nodes: {G.number_of_nodes()}")
    logger.info(f"  Edges: {G.number_of_edges()}")
    logger.info(f"  Density: {nx.density(G_simple):.4f}")
    
    if nx.is_weakly_connected(G):
        logger.info(f"  Connected: Yes (single component)")
    else:
        components = list(nx.weakly_connected_components(G))
        logger.info(f"  Connected: No ({len(components)} components)")
    
    logger.info(f"\n  Node types:")
    for ntype, count in sorted(node_types.items(), key=lambda x: -x[1]):
        logger.info(f"    {ntype}: {count}")
    
    logger.info(f"\n  Edge relations:")
    for rel, count in sorted(edge_relations.items(), key=lambda x: -x[1]):
        logger.info(f"    {rel}: {count}")
    
    if years_on_edges:
        logger.info(f"\n  Temporal range: {min(years_on_edges)} - {max(years_on_edges)}")
    
    logger.info(f"\n  Saved to: {graphml_path}")
    logger.info(f"  Pickle: {pickle_path}")
    
    return G


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    build_knowledge_graph(config)
