"""
Stage 6: Visualisation

Generates interactive knowledge graph visualisations and
gap analysis charts.

Usage:
    python run_pipeline.py --stage visualise
"""

import pickle
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from pyvis.network import Network
from src.utils import get_logger, save_json, load_json, ensure_dir

logger = get_logger("visualise")

# Color map for entity types
TYPE_COLORS = {
    "METHOD": "#7F77DD",
    "DATASET": "#1D9E75",
    "METRIC": "#D85A30",
    "CONCEPT": "#378ADD",
    "FINDING": "#D4537E",
    "TOOL": "#639922",
    "UNKNOWN": "#888780",
}


def create_interactive_graph(G, gaps, output_path):
    """Create an interactive pyvis graph with gaps highlighted."""
    logger.info("Creating interactive graph visualisation...")
    
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200)
    
    # Collect gap-related nodes
    gap_nodes = set()
    for gap in gaps:
        if gap["type"] == "missing_link":
            gap_nodes.add(gap.get("head", ""))
            gap_nodes.add(gap.get("tail", ""))
        elif gap["type"] == "orphan_cluster":
            for member in gap.get("members", [])[:10]:
                gap_nodes.add(member)
        elif gap["type"] == "temporal_decay":
            gap_nodes.add(gap.get("concept", ""))
    
    # Add nodes
    for node, data in G.nodes(data=True):
        ntype = data.get("type", "UNKNOWN")
        color = TYPE_COLORS.get(ntype, "#888780")
        centrality = data.get("degree_centrality", 0.01)
        size = max(10, min(50, centrality * 500))
        
        # Highlight gap nodes with red border
        border_color = "#E24B4A" if node in gap_nodes else color
        border_width = 3 if node in gap_nodes else 1
        
        papers = data.get("papers", [])
        if isinstance(papers, str):
            papers = papers.split(",")
        
        title = f"<b>{node}</b><br>Type: {ntype}<br>Centrality: {centrality:.3f}<br>Papers: {len(papers)}"
        
        net.add_node(
            node,
            label=node if centrality > 0.02 else "",
            title=title,
            color={"background": color, "border": border_color},
            borderWidth=border_width,
            size=size,
        )
    
    # Add edges
    added_edges = set()
    for u, v, data in G.edges(data=True):
        edge_key = (u, v)
        if edge_key in added_edges:
            continue
        added_edges.add(edge_key)
        
        relation = data.get("relation", "")
        year = data.get("year", "")
        
        net.add_edge(
            u, v,
            title=f"{relation} ({year})",
            color="#CCCCCC",
            width=0.5,
        )
    
    # Add missing link gaps as dashed red edges
    for gap in gaps:
        if gap["type"] == "missing_link":
            head = gap.get("head", "")
            tail = gap.get("tail", "")
            if G.has_node(head) and G.has_node(tail):
                net.add_edge(
                    head, tail,
                    title=f"PREDICTED GAP: {gap.get('relation', '')} (score: {gap.get('composite_score', 0):.3f})",
                    color="#E24B4A",
                    width=2,
                    dashes=True,
                )
    
    net.save_graph(str(output_path))
    logger.info(f"  Saved interactive graph: {output_path}")


def plot_gap_distribution(gaps, figures_dir):
    """Create charts showing gap type distribution and scores."""
    logger.info("Creating gap analysis charts...")
    
    # Gap type distribution
    type_counts = defaultdict(int)
    for gap in gaps:
        type_counts[gap["type"]] += 1
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Gap type pie chart
    ax = axes[0]
    labels = list(type_counts.keys())
    sizes = list(type_counts.values())
    colors = ["#7F77DD", "#1D9E75", "#D85A30"][:len(labels)]
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
    ax.set_title("Gap Types Distribution")
    
    # 2. Score distribution
    ax = axes[1]
    scores = [g.get("composite_score", 0) for g in gaps]
    ax.hist(scores, bins=20, color="#378ADD", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Composite Score")
    ax.set_ylabel("Count")
    ax.set_title("Gap Score Distribution")
    
    # 3. Top gaps bar chart
    ax = axes[2]
    top_10 = gaps[:10]
    names = [f"#{g['rank']}" for g in top_10]
    top_scores = [g.get("composite_score", 0) for g in top_10]
    bar_colors = []
    for g in top_10:
        if g["type"] == "missing_link":
            bar_colors.append("#7F77DD")
        elif g["type"] == "orphan_cluster":
            bar_colors.append("#1D9E75")
        else:
            bar_colors.append("#D85A30")
    
    ax.barh(names[::-1], top_scores[::-1], color=bar_colors[::-1], edgecolor="white")
    ax.set_xlabel("Composite Score")
    ax.set_title("Top 10 Gaps by Score")
    
    plt.tight_layout()
    plt.savefig(figures_dir / "gap_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {figures_dir / 'gap_analysis.png'}")


def plot_temporal_profile(gaps, figures_dir):
    """Plot temporal decay profiles for decaying concepts."""
    decay_gaps = [g for g in gaps if g["type"] == "temporal_decay"]
    
    if not decay_gaps:
        return
    
    top_decay = decay_gaps[:6]  # Top 6 decaying concepts
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, gap in enumerate(top_decay):
        if i >= len(axes):
            break
        ax = axes[i]
        profile = gap.get("temporal_profile", {})
        years = sorted(profile.keys(), key=int)
        counts = [profile[y] for y in years]
        
        ax.bar([str(y) for y in years], counts, color="#D85A30", edgecolor="white", alpha=0.8)
        ax.set_title(gap.get("concept", "")[:30], fontsize=10)
        ax.set_ylabel("Activity")
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Hide unused axes
    for j in range(len(top_decay), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle("Temporal Decay Profiles - Top Declining Concepts", fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / "temporal_decay.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {figures_dir / 'temporal_decay.png'}")


def plot_graph_stats(G, figures_dir):
    """Plot basic knowledge graph statistics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Node type distribution
    ax = axes[0]
    type_counts = defaultdict(int)
    for _, data in G.nodes(data=True):
        type_counts[data.get("type", "UNKNOWN")] += 1
    
    types = list(type_counts.keys())
    counts = list(type_counts.values())
    colors = [TYPE_COLORS.get(t, "#888780") for t in types]
    ax.barh(types, counts, color=colors, edgecolor="white")
    ax.set_xlabel("Count")
    ax.set_title("Entity Types")
    
    # 2. Degree distribution
    ax = axes[1]
    G_simple = nx.DiGraph(G)
    degrees = [d for _, d in G_simple.degree()]
    ax.hist(degrees, bins=30, color="#378ADD", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title("Degree Distribution")
    ax.set_yscale("log")
    
    # 3. Edge relation distribution
    ax = axes[2]
    rel_counts = defaultdict(int)
    for _, _, data in G.edges(data=True):
        rel_counts[data.get("relation", "UNKNOWN")] += 1
    
    rels = sorted(rel_counts.keys(), key=lambda r: rel_counts[r], reverse=True)[:10]
    rcounts = [rel_counts[r] for r in rels]
    ax.barh(rels[::-1], rcounts[::-1], color="#7F77DD", edgecolor="white")
    ax.set_xlabel("Count")
    ax.set_title("Top Relationship Types")
    
    plt.tight_layout()
    plt.savefig(figures_dir / "graph_stats.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {figures_dir / 'graph_stats.png'}")


def generate_visualisations(config):
    """Main visualisation function."""
    graph_dir = Path(config["paths"]["graph"])
    output_dir = ensure_dir(config["paths"]["outputs"])
    figures_dir = ensure_dir(config["paths"]["figures"])
    
    # Load graph
    pkl_path = graph_dir / "knowledge_graph.pkl"
    if not pkl_path.exists():
        logger.error(f"Graph not found: {pkl_path}")
        return
    
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    
    # Load scored gaps
    gaps_path = output_dir / "gaps_ranked_top.json"
    gaps = load_json(gaps_path) if gaps_path.exists() else []
    
    # --- Generate visualisations ---
    
    # 1. Interactive graph
    create_interactive_graph(G, gaps, output_dir / "graph_viz.html")
    
    # 2. Graph statistics
    plot_graph_stats(G, figures_dir)
    
    # 3. Gap analysis
    if gaps:
        plot_gap_distribution(gaps, figures_dir)
        plot_temporal_profile(gaps, figures_dir)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"  VISUALISATION COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"  Interactive graph: {output_dir / 'graph_viz.html'}")
    logger.info(f"  Charts: {figures_dir}")


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    generate_visualisations(config)
