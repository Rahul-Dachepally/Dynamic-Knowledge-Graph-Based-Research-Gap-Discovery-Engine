"""
KG Gap Discovery Engine - Streamlit Frontend
Run: streamlit run app.py
"""

import streamlit as st
import yaml
import json
import time
import threading
import queue
import sys
import os
import pickle
import pandas as pd
import networkx as nx
from pathlib import Path
from datetime import datetime


import warnings

from src.visualise import generate_visualisations
warnings.filterwarnings("ignore", message="Accessing `__path__`")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="KG Research Gap Discovery",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
.stProgress > div > div > div > div { background-color: #7F77DD; }
.gap-card {
    background: #f8f9fa;
    border-left: 4px solid #7F77DD;
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 10px;
}
.gap-card.missing { border-left-color: #7F77DD; }
.gap-card.orphan  { border-left-color: #1D9E75; }
.gap-card.decay   { border-left-color: #D85A30; }
.metric-box {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────

def load_base_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def build_run_config(base_config, topic, num_papers, groq_key):
    """Override base config with user inputs."""
    cfg = base_config.copy()
    import copy
    cfg = copy.deepcopy(base_config)

    cfg["project"]["domain"] = topic
    cfg["api_keys"]["groq"] = groq_key
    cfg["collection"]["max_papers"] = num_papers
    cfg["filtering"]["target_corpus_size"] = min(num_papers, 150)

    # Generate search queries from topic using simple heuristic
    # (replaced by LLM query agent below if key present)
    words = topic.lower().split()
    cfg["collection"]["queries"] = [
        topic,
        " ".join(words[:3]) if len(words) >= 3 else topic,
        f"{topic} survey",
        f"{topic} framework",
        f"{topic} deep learning",
    ]

    # Use a unique run folder so multiple topics don't overwrite each other
    slug = topic.lower().replace(" ", "_")[:30]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{slug}_{ts}"

    cfg["paths"] = {
        "raw_data":       f"runs/{run_id}/data/raw",
        "processed_data": f"runs/{run_id}/data/processed",
        "triples":        f"runs/{run_id}/data/triples",
        "graph":          f"runs/{run_id}/data/graph",
        "outputs":        f"runs/{run_id}/outputs",
        "figures":        f"runs/{run_id}/outputs/figures",
        "prompts":        "prompts",
    }

    return cfg, run_id


def generate_queries_with_llm(topic, groq_key):
    """Use Groq to generate smart search queries for the topic."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"""Generate 5 academic search queries for finding papers about: "{topic}"
Return ONLY a JSON array of 5 short search strings (2-5 words each), no explanation.
Example: ["query one", "query two", "query three", "query four", "query five"]"""
            }],
            temperature=0.3,
            max_tokens=200,
        )
        content = resp.choices[0].message.content.strip()
        content = content.lstrip("```json").lstrip("```").rstrip("```").strip()
        queries = json.loads(content)
        if isinstance(queries, list) and len(queries) >= 3:
            return queries[:5]
    except Exception:
        pass
    # Fallback
    return [topic, f"{topic} survey", f"{topic} framework",
            f"{topic} deep learning", f"{topic} methods"]


def run_pipeline_with_progress(cfg, progress_queue):
    """Run the full pipeline in a background thread, posting progress."""
    try:
        # Ensure src is importable
        sys.path.insert(0, str(Path(__file__).parent))

        progress_queue.put(("status", "🔍 Generating search queries..."))
        progress_queue.put(("progress", 5))

        from src.collect import collect_papers
        progress_queue.put(("status", "📥 Collecting papers from Semantic Scholar..."))
        progress_queue.put(("progress", 10))
        collect_papers(cfg)
        progress_queue.put(("progress", 25))

        from src.filter import filter_corpus
        progress_queue.put(("status", "🔎 Screening papers for relevance..."))
        progress_queue.put(("progress", 30))
        filter_corpus(cfg)
        progress_queue.put(("progress", 45))

        from src.extract_triples import extract_all_triples
        progress_queue.put(("status", "🧠 Extracting knowledge triples..."))
        progress_queue.put(("progress", 50))
        extract_all_triples(cfg)
        progress_queue.put(("progress", 65))

        from src.build_graph import build_knowledge_graph
        progress_queue.put(("status", "🕸️ Building knowledge graph..."))
        progress_queue.put(("progress", 70))
        build_knowledge_graph(cfg)
        progress_queue.put(("progress", 80))

        from src.detect_gaps import detect_all_gaps
        progress_queue.put(("status", "🔬 Detecting research gaps..."))
        progress_queue.put(("progress", 85))
        detect_all_gaps(cfg)
        progress_queue.put(("progress", 92))

        from src.score_gaps import score_and_rank_gaps
        progress_queue.put(("status", "📊 Scoring and ranking gaps..."))
        progress_queue.put(("progress", 96))
        score_and_rank_gaps(cfg)
        progress_queue.put(("progress", 100))

        from src.visualise import generate_visualisations
        progress_queue.put(("status", "🎨 Generating visualisations..."))
        generate_visualisations(cfg)
        progress_queue.put(("progress", 100))

        progress_queue.put(("done", cfg))

    except Exception as e:
        import traceback
        progress_queue.put(("error", f"{e}\n\n{traceback.format_exc()}"))


def load_results(cfg):
    """Load all result files for display."""
    out = cfg["paths"]["outputs"]
    results = {}

    gaps_path = Path(out) / "gaps_ranked_top.json"
    if gaps_path.exists():
        with open(gaps_path) as f:
            results["gaps"] = json.load(f)

    csv_path = Path(out) / "gaps_ranked.csv"
    if csv_path.exists():
        results["gaps_df"] = pd.read_csv(csv_path)

    graph_path = Path(cfg["paths"]["graph"]) / "knowledge_graph.pkl"
    if graph_path.exists():
        with open(graph_path, "rb") as f:
            results["graph"] = pickle.load(f)

    html_path = Path(out) / "graph_viz.html"
    if html_path.exists():
        with open(html_path) as f:
            results["graph_html"] = f.read()

    return results


# ── Sidebar ─────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/graph.png", width=60)
    st.title("KG Gap Discovery")
    st.caption("Temporal Knowledge Graph-Based Research Gap Detection")
    st.divider()

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )

    st.divider()
    st.markdown("**Pipeline stages:**")
    st.markdown("1. 📥 Collect papers")
    st.markdown("2. 🔎 Filter corpus")
    st.markdown("3. 🧠 Extract triples")
    st.markdown("4. 🕸️ Build KG")
    st.markdown("5. 🔬 Detect gaps")
    st.markdown("6. 📊 Score & rank")

    st.divider()
    st.caption("Based on Mulla et al. (2026) — HCAIep '26")


# ── Main area ───────────────────────────────────────────────

st.title("🔬 Research Gap Discovery Engine")
st.markdown("Enter any research topic and let the knowledge graph find what the field is missing.")

# ── Input form ──────────────────────────────────────────────

col1, col2 = st.columns([3, 1])

with col1:
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. federated learning privacy, medical image segmentation, ...",
        help="Be specific — this becomes the domain for the entire pipeline",
    )

with col2:
    num_papers = st.slider(
        "Max papers",
        min_value=20,
        max_value=150,
        value=50,
        step=10,
        help="More papers = better gaps but longer runtime",
    )

run_button = st.button(
    "🚀 Discover Gaps",
    type="primary",
    disabled=not (topic and groq_key),
    width="stretch",
)

if not groq_key:
    st.info("Add your Groq API key in the sidebar to get started.")

# ── Pipeline execution ──────────────────────────────────────

if run_button and topic and groq_key:

    base_cfg = load_base_config()

    with st.spinner("Generating optimised search queries..."):
        queries = generate_queries_with_llm(topic, groq_key)

    cfg, run_id = build_run_config(base_cfg, topic, num_papers, groq_key)
    cfg["collection"]["queries"] = queries

    st.markdown(f"**Search queries generated:** `{'` · `'.join(queries)}`")

    # Progress UI
    status_box   = st.empty()
    progress_bar = st.progress(0)
    log_box      = st.empty()

    progress_queue = queue.Queue()

    thread = threading.Thread(
        target=run_pipeline_with_progress,
        args=(cfg, progress_queue),
        daemon=True,
    )
    thread.start()

    done_cfg = None
    error    = None
    logs     = []

    while thread.is_alive() or not progress_queue.empty():
        try:
            msg_type, payload = progress_queue.get(timeout=0.5)
            if msg_type == "status":
                status_box.markdown(f"**{payload}**")
                logs.append(payload)
                log_box.markdown("\n".join(f"- {l}" for l in logs))
            elif msg_type == "progress":
                progress_bar.progress(payload)
            elif msg_type == "done":
                done_cfg = payload
            elif msg_type == "error":
                error = payload
        except queue.Empty:
            continue

    thread.join()

    if error:
        st.error(f"Pipeline failed:\n\n```\n{error}\n```")
        st.stop()

    if done_cfg:
        st.success("✅ Pipeline complete!")
        st.session_state["results"] = load_results(done_cfg)
        st.session_state["run_cfg"] = done_cfg
        st.session_state["topic"]   = topic


# ── Results display ─────────────────────────────────────────

if "results" in st.session_state:
    results = st.session_state["results"]
    cfg     = st.session_state["run_cfg"]
    topic   = st.session_state.get("topic", "")

    st.divider()
    st.header(f"Results: {topic}")

    # ── Metric cards ──
    gaps    = results.get("gaps", [])
    G       = results.get("graph")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Gaps Found", len(gaps))
    with col2:
        ml = sum(1 for g in gaps if g["type"] == "missing_link")
        st.metric("Missing Links", ml)
    with col3:
        oc = sum(1 for g in gaps if g["type"] == "orphan_cluster")
        st.metric("Orphan Clusters", oc)
    with col4:
        td = sum(1 for g in gaps if g["type"] == "temporal_decay")
        st.metric("Decaying Concepts", td)

    if G:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("KG Nodes", G.number_of_nodes())
        with col2:
            st.metric("KG Edges", G.number_of_edges())
        with col3:
            comps = nx.number_weakly_connected_components(G)
            st.metric("Components", comps)

    st.divider()

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["📋 Ranked Gaps", "🕸️ Knowledge Graph", "📈 Analytics"])

    # ── Tab 1: Ranked gaps ──
    with tab1:
        st.subheader("Top Research Gaps")

        type_filter = st.multiselect(
            "Filter by type",
            ["missing_link", "orphan_cluster", "temporal_decay"],
            default=["missing_link", "orphan_cluster", "temporal_decay"],
        )

        filtered = [g for g in gaps if g["type"] in type_filter]

        TYPE_COLORS = {
            "missing_link":   ("🔗", "missing", "Missing Link"),
            "orphan_cluster": ("🏝️", "orphan",  "Orphan Cluster"),
            "temporal_decay": ("📉", "decay",   "Temporal Decay"),
        }

        for g in filtered[:30]:
            icon, css_class, label = TYPE_COLORS.get(g["type"], ("❓", "", g["type"]))
            score = g.get("composite_score", 0)
            desc  = g.get("description", "")

            st.markdown(f"""
<div class="gap-card {css_class}">
  <strong>#{g['rank']} {icon} {label}</strong>
  &nbsp;&nbsp;<code>score: {score:.4f}</code><br>
  <span style="color:#444">{desc}</span>
</div>
""", unsafe_allow_html=True)

        # Download button
        if "gaps_df" in results:
            csv = results["gaps_df"].to_csv(index=False)
            st.download_button(
                "⬇️ Download gaps as CSV",
                csv,
                file_name=f"gaps_{topic.replace(' ','_')}.csv",
                mime="text/csv",
            )

    # ── Tab 2: Knowledge graph ──
    with tab2:
        st.subheader("Interactive Knowledge Graph")
        st.caption("Nodes = research concepts. Red borders = gap-related nodes. Hover for details.")

        graph_html = results.get("graph_html")
        if graph_html:
            st.components.v1.html(graph_html, height=650, scrolling=False)
        else:
            st.warning("Graph visualisation not available. Run the visualise stage.")

        # Graph stats table
        if G:
            st.markdown("**Top 10 most connected concepts:**")
            G_simple = nx.DiGraph(G)
            degree_cent = nx.degree_centrality(G_simple)
            top_nodes = sorted(degree_cent.items(), key=lambda x: -x[1])[:10]
            df_nodes = pd.DataFrame(top_nodes, columns=["Concept", "Centrality"])
            df_nodes["Type"] = df_nodes["Concept"].apply(
                lambda n: G.nodes[n].get("type", "?") if G.has_node(n) else "?"
            )
            st.dataframe(df_nodes, width="stretch")

    # ── Tab 3: Analytics ──
    with tab3:
        st.subheader("Gap Analytics")

        fig_path = Path(cfg["paths"]["figures"]) / "gap_analysis.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Gap distribution and scores")

        decay_path = Path(cfg["paths"]["figures"]) / "temporal_decay.png"
        if decay_path.exists():
            st.image(str(decay_path), caption="Temporal decay profiles")

        stats_path = Path(cfg["paths"]["figures"]) / "graph_stats.png"
        if stats_path.exists():
            st.image(str(stats_path), caption="Knowledge graph statistics")

        # Edge relation breakdown
        if G:
            rel_counts = {}
            for _, _, d in G.edges(data=True):
                r = d.get("relation", "?")
                rel_counts[r] = rel_counts.get(r, 0) + 1
            df_rel = pd.DataFrame(
                sorted(rel_counts.items(), key=lambda x: -x[1]),
                columns=["Relation", "Count"]
            )
            st.markdown("**Edge relation distribution:**")
            st.bar_chart(df_rel.set_index("Relation"))
