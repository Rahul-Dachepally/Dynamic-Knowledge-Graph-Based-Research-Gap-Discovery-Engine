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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

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

# Nodes whose labels should never appear in the paper figure
BAD_LABELS = {
    "proposed method", "our method", "the model", "this paper",
    "our approach", "the proposed method", "this work", "our framework",
    "proposed approach", "method",
}


def _safe_profile(gap, G):
    """
    Return {year_int: count} for a temporal_decay gap.

    Tries three sources in order:
      1. gap['temporal_profile'] with int keys
      2. gap['temporal_profile'] with str keys (JSON round-trips them)
      3. Count edges on the concept node directly from G
    """
    profile = gap.get("temporal_profile", {})

    # Source 1 & 2 — stored profile
    if profile:
        # normalise keys to int
        try:
            return {int(k): int(v) for k, v in profile.items()}
        except (ValueError, TypeError):
            pass

    # Source 3 — recompute from graph edges
    concept = gap.get("concept", "")
    if not concept or not G.has_node(concept):
        return {}

    year_counts = defaultdict(int)
    for u, v, data in G.edges(data=True):
        if u == concept or v == concept:
            y = data.get("year")
            if y:
                try:
                    year_counts[int(y)] += 1
                except (ValueError, TypeError):
                    pass

    return dict(year_counts)


def generate_paper_figure(G, gaps, figures_dir):
    """
    Publication-quality three-panel KG figure.

    Panel left  : full knowledge graph (main + gap nodes highlighted)
    Panel top-r : zoom of the largest orphan cluster
    Panel bot-r : temporal decay bar chart for top-3 decaying concepts
    """
    try:
        import community as community_louvain
    except ImportError:
        import community as community_louvain  # python-louvain

    logger_print = print   # use print so it works even if logger not imported here

    fig = plt.figure(figsize=(18, 10), facecolor="white")
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.45, wspace=0.30,
        left=0.04, right=0.97, top=0.93, bottom=0.07,
    )
    ax_main  = fig.add_subplot(gs[:, :2])   # left 2/3
    ax_zoom  = fig.add_subplot(gs[0, 2])    # top-right
    ax_decay = fig.add_subplot(gs[1, 2])    # bottom-right

    # ── collect gap nodes ────────────────────────────────────────
    gap_nodes = set()
    for gap in gaps:
        if gap["type"] == "orphan_cluster":
            gap_nodes.update(gap.get("members", [])[:15])
        elif gap["type"] == "temporal_decay":
            gap_nodes.add(gap.get("concept", ""))
        elif gap["type"] == "missing_link":
            gap_nodes.add(gap.get("head", ""))
            gap_nodes.add(gap.get("tail", ""))
    gap_nodes.discard("")

    # ── build simple DiGraph for layout ─────────────────────────
    G_simple = nx.DiGraph(G)

    # ── layout ──────────────────────────────────────────────────
    try:
        # kamada_kawai gives nicer spacing when graph is not too sparse
        if G_simple.number_of_edges() > 10:
            pos = nx.kamada_kawai_layout(G_simple)
        else:
            pos = nx.spring_layout(G_simple, seed=42, k=2.8)
    except Exception:
        pos = nx.spring_layout(G_simple, seed=42, k=2.8)

    # ── node styling ─────────────────────────────────────────────
    node_list   = list(G_simple.nodes())
    node_colors = [TYPE_COLORS.get(G.nodes[n].get("type", "UNKNOWN"), "#888") for n in node_list]
    node_sizes  = []
    for n in node_list:
        cent = G.nodes[n].get("degree_centrality", 0.01)
        base = 300 if n in gap_nodes else 120
        node_sizes.append(max(base, min(900, cent * 4000)))

    # ── draw edges ───────────────────────────────────────────────
    nx.draw_networkx_edges(
        G_simple, pos, ax=ax_main,
        edge_color="#CCCCCC", width=0.7, alpha=0.5,
        arrows=True, arrowsize=10,
        connectionstyle="arc3,rad=0.05",
    )

    # ── draw normal nodes ─────────────────────────────────────────
    nx.draw_networkx_nodes(
        G_simple, pos, ax=ax_main,
        node_color=node_colors, node_size=node_sizes, alpha=0.92,
    )

    # ── red ring around gap nodes ─────────────────────────────────
    gap_present = [n for n in node_list if n in gap_nodes]
    if gap_present:
        gap_sizes = [node_sizes[node_list.index(n)] + 120 for n in gap_present]
        nx.draw_networkx_nodes(
            G_simple, pos, ax=ax_main,
            nodelist=gap_present,
            node_color="none", node_size=gap_sizes,
            edgecolors="#E24B4A", linewidths=2.2,
        )

    # ── labels: only high-centrality and non-junk nodes ──────────
    label_nodes = {
        n: n for n in node_list
        if (G.nodes[n].get("degree_centrality", 0) > 0.04
            or n in gap_nodes)
        and n.lower() not in BAD_LABELS
    }
    # truncate long labels
    label_nodes = {
        n: (lab[:24] + "…" if len(lab) > 24 else lab)
        for n, lab in label_nodes.items()
    }
    nx.draw_networkx_labels(
        G_simple, pos, labels=label_nodes, ax=ax_main,
        font_size=6.5, font_color="#1F2937",
    )

    ax_main.set_title(
        f"Full knowledge graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
        fontsize=11, pad=8,
    )
    ax_main.axis("off")

    # ── legend ───────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=v, label=k)
        for k, v in TYPE_COLORS.items() if k != "UNKNOWN"
    ]
    handles.append(
        mpatches.Patch(
            edgecolor="#E24B4A", facecolor="none",
            label="Gap node", linewidth=2,
        )
    )
    ax_main.legend(
        handles=handles, loc="lower left", fontsize=7,
        ncol=2, framealpha=0.85,
    )

    # ── Panel 2: Orphan cluster zoom ─────────────────────────────
    orphan_gaps = [g for g in gaps if g["type"] == "orphan_cluster"]
    if orphan_gaps:
        biggest = max(orphan_gaps, key=lambda x: x.get("size", 0))
        members = [m for m in biggest.get("members", []) if G_simple.has_node(m)]

        if len(members) >= 2:
            sub     = G_simple.subgraph(members)
            sub_pos = nx.spring_layout(sub, seed=7, k=3.0)

            sub_colors = [
                TYPE_COLORS.get(G.nodes[n].get("type", "UNKNOWN"), "#888")
                for n in sub.nodes()
            ]
            nx.draw_networkx_edges(
                sub, sub_pos, ax=ax_zoom,
                edge_color="#AAAAAA", width=1.0, alpha=0.7,
                arrows=True, arrowsize=10,
            )
            nx.draw_networkx_nodes(
                sub, sub_pos, ax=ax_zoom,
                node_color=sub_colors, node_size=220, alpha=0.92,
            )
            # truncated labels to avoid clipping
            sub_labels = {
                n: (n[:20] + "…" if len(n) > 20 else n)
                for n in sub.nodes()
            }
            nx.draw_networkx_labels(
                sub, sub_pos, labels=sub_labels, ax=ax_zoom,
                font_size=6.5, font_color="#1F2937", clip_on=False,
            )
            ax_zoom.set_title(
                f"Orphan cluster #{biggest['community_id']}\n"
                f"({biggest['size']} nodes, "
                f"{biggest['inter_edge_ratio']:.0%} bridge edges)",
                fontsize=9, pad=6,
            )
        else:
            ax_zoom.text(
                0.5, 0.5, "Cluster too small\nto visualise",
                ha="center", va="center", transform=ax_zoom.transAxes,
                fontsize=9, color="#6B7280",
            )
            ax_zoom.set_title("Orphan cluster", fontsize=9)
    else:
        ax_zoom.text(
            0.5, 0.5, "No orphan clusters\ndetected",
            ha="center", va="center", transform=ax_zoom.transAxes,
            fontsize=9, color="#6B7280",
        )
        ax_zoom.set_title("Orphan cluster", fontsize=9)

    ax_zoom.axis("off")

    # ── Panel 3: Temporal decay bar chart ────────────────────────
    decay_gaps = sorted(
        [g for g in gaps if g["type"] == "temporal_decay"],
        key=lambda x: x.get("decay_rate", 0),
        reverse=True,
    )[:3]

    if decay_gaps:
        # collect all years present across the three concepts
        all_profiles = [_safe_profile(g, G) for g in decay_gaps]
        all_years    = sorted({y for p in all_profiles for y in p.keys()})

        if all_years:
            bar_w   = 0.25
            colors  = ["#D85A30", "#BA7517", "#7F77DD"]
            xs_base = list(range(len(all_years)))

            plotted = 0
            for j, (dg, profile) in enumerate(zip(decay_gaps, all_profiles)):
                if not profile:
                    continue
                vals = [profile.get(y, 0) for y in all_years]
                xs   = [x + j * bar_w for x in xs_base]
                ax_decay.bar(
                    xs, vals, width=bar_w,
                    color=colors[j % len(colors)], alpha=0.85,
                    label=dg["concept"][:22],
                )
                plotted += 1

            if plotted > 0:
                tick_xs = [x + bar_w for x in xs_base]
                ax_decay.set_xticks(tick_xs)
                ax_decay.set_xticklabels(
                    [str(y) for y in all_years],
                    fontsize=7, rotation=45, ha="right",
                )
                ax_decay.set_ylabel("Edge count", fontsize=8)
                ax_decay.set_title(
                    "Temporal decay — top stalled concepts", fontsize=9, pad=6,
                )
                ax_decay.legend(fontsize=7, loc="upper right")
                ax_decay.tick_params(axis="y", labelsize=7)
                ax_decay.yaxis.get_major_locator().set_params(integer=True)
            else:
                ax_decay.text(
                    0.5, 0.5,
                    "Temporal profiles not stored\n(run pipeline to regenerate)",
                    ha="center", va="center", transform=ax_decay.transAxes,
                    fontsize=8, color="#6B7280",
                )
                ax_decay.set_title("Temporal decay", fontsize=9)
        else:
            ax_decay.text(
                0.5, 0.5,
                "No year data on edges\n(check source_year in triples)",
                ha="center", va="center", transform=ax_decay.transAxes,
                fontsize=8, color="#6B7280",
            )
            ax_decay.set_title("Temporal decay", fontsize=9)
    else:
        ax_decay.text(
            0.5, 0.5, "No temporal decay\ngaps detected",
            ha="center", va="center", transform=ax_decay.transAxes,
            fontsize=9, color="#6B7280",
        )
        ax_decay.set_title("Temporal decay", fontsize=9)

    # ── save ─────────────────────────────────────────────────────
    out_path = Path(figures_dir) / "figure4_kg_publication.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger_print(f"  Saved publication figure: {out_path}")
    return out_path





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
    
    generate_paper_figure(G, gaps, figures_dir)
    
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
