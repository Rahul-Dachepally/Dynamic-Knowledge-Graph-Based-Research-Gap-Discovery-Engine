"""
KG Gap Discovery Engine - Streamlit Frontend
Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`")

import streamlit as st
import yaml
import json
import time
import threading
import queue
import sys
import os
import pickle
import logging
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from datetime import datetime

logging.getLogger("transformers").setLevel(logging.ERROR)

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="KG Research Gap Discovery",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
.gap-card {
    background: #1e1e2e;
    border-left: 4px solid #7F77DD;
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 10px;
    color: #ffffff !important;
}
.gap-card.missing { border-left-color: #7F77DD; }
.gap-card.orphan  { border-left-color: #1D9E75; }
.gap-card.decay   { border-left-color: #D85A30; }
.gap-card.mulla   { border-left-color: #F0A500; }
.gap-card.simple  { border-left-color: #888780; }
.gap-card strong  { color: #ffffff !important; }
.gap-card small   { color: #cccccc !important; }
.gap-card code    { background: #333; color: #adf; padding: 2px 6px; border-radius: 3px; }
.method-header {
    font-size: 1.1em;
    font-weight: bold;
    padding: 8px 0;
    margin-bottom: 8px;
    border-bottom: 2px solid #444;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────

def load_base_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def build_run_config(base_config, topic, num_papers, groq_key):
    import copy
    cfg = copy.deepcopy(base_config)
    cfg["project"]["domain"]               = topic
    cfg["api_keys"]["groq"]                = groq_key
    cfg["collection"]["max_papers"]        = num_papers
    cfg["filtering"]["target_corpus_size"] = min(num_papers, 150)

    slug   = topic.lower().replace(" ", "_")[:30]
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    try:
        from openai import OpenAI
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content":
                f"""Generate 5 academic search queries for: "{topic}"
Return ONLY a JSON array of 5 short search strings (2-5 words each).
Example: ["query one", "query two", "query three", "query four", "query five"]"""}],
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
    return [topic, f"{topic} survey", f"{topic} framework",
            f"{topic} deep learning", f"{topic} methods"]


def run_pipeline_with_progress(cfg, progress_queue):
    try:
        sys.path.insert(0, str(Path(__file__).parent))

        progress_queue.put(("status", "📥 Collecting papers from Semantic Scholar..."))
        progress_queue.put(("progress", 8))
        from src.collect import collect_papers
        collect_papers(cfg)
        progress_queue.put(("progress", 20))

        progress_queue.put(("status", "🔎 Screening papers for relevance..."))
        progress_queue.put(("progress", 22))
        from src.filter import filter_corpus
        filter_corpus(cfg)
        progress_queue.put(("progress", 40))

        progress_queue.put(("status", "🧠 Extracting knowledge triples..."))
        progress_queue.put(("progress", 42))
        from src.extract_triples import extract_all_triples
        extract_all_triples(cfg)
        progress_queue.put(("progress", 58))

        progress_queue.put(("status", "🕸️ Building knowledge graph..."))
        progress_queue.put(("progress", 60))
        from src.build_graph import build_knowledge_graph
        build_knowledge_graph(cfg)
        progress_queue.put(("progress", 72))

        progress_queue.put(("status", "🔬 Detecting research gaps..."))
        progress_queue.put(("progress", 74))
        from src.detect_gaps import detect_all_gaps
        detect_all_gaps(cfg)
        progress_queue.put(("progress", 84))

        progress_queue.put(("status", "📊 Scoring and ranking gaps..."))
        progress_queue.put(("progress", 86))
        from src.score_gaps import score_and_rank_gaps
        score_and_rank_gaps(cfg)
        progress_queue.put(("progress", 92))

        progress_queue.put(("status", "🎨 Generating visualisations..."))
        from src.visualise import generate_visualisations
        generate_visualisations(cfg)
        progress_queue.put(("progress", 100))

        progress_queue.put(("done", cfg))

    except Exception as e:
        import traceback
        progress_queue.put(("error", f"{e}\n\n{traceback.format_exc()}"))


def run_rag_with_progress(cfg, progress_queue):
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.rag_baseline import run_rag_baseline

        progress_queue.put(("status", "📚 Embedding abstracts for retrieval..."))
        progress_queue.put(("progress", 10))
        progress_queue.put(("status", "🤖 Running Mulla et al. RAG baseline..."))
        progress_queue.put(("progress", 20))

        results = run_rag_baseline(cfg)

        progress_queue.put(("progress", 100))
        progress_queue.put(("done", results))

    except Exception as e:
        import traceback
        progress_queue.put(("error", f"{e}\n\n{traceback.format_exc()}"))


def load_results(cfg):
    out     = cfg["paths"]["outputs"]
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

    mulla_path = Path(out) / "rag_mulla_gaps.json"
    if mulla_path.exists():
        with open(mulla_path) as f:
            results["mulla_gaps"] = json.load(f)

    simple_path = Path(out) / "rag_simple_gaps.json"
    if simple_path.exists():
        with open(simple_path) as f:
            results["simple_gaps"] = json.load(f)

    metrics_path = Path(out) / "comparison_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            results["comparison_metrics"] = json.load(f)

    return results


# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 KG Gap Discovery")
    st.caption("Temporal Knowledge Graph-Based Research Gap Detection")
    st.divider()

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free key at console.groq.com",
    )

    st.divider()
    st.markdown("**Pipeline stages:**")
    for stage in ["📥 Collect papers", "🔎 Filter corpus", "🧠 Extract triples",
                  "🕸️ Build KG", "🔬 Detect gaps", "📊 Score & rank",
                  "🎨 Visualise", "⚖️ RAG comparison"]:
        st.markdown(f"- {stage}")
    st.divider()
    st.caption("Based on Mulla et al. (2026) — HCAIep '26")


# ── Main ────────────────────────────────────────────────────
st.title("🔬 Research Gap Discovery Engine")
st.markdown("Enter any research topic — the knowledge graph finds what the field is missing.")

col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. federated learning privacy, medical image segmentation...",
    )
with col2:
    num_papers = st.slider("Max papers", 20, 150, 50, 10)

run_button = st.button(
    "🚀 Discover Gaps",
    type="primary",
    disabled=not (topic and groq_key),
)

if not groq_key:
    st.info("Add your Groq API key in the sidebar to get started.")

# ── Pipeline execution ───────────────────────────────────────
if run_button and topic and groq_key:
    base_cfg = load_base_config()

    with st.spinner("Generating search queries..."):
        queries = generate_queries_with_llm(topic, groq_key)

    cfg, run_id = build_run_config(base_cfg, topic, num_papers, groq_key)
    cfg["collection"]["queries"] = queries

    st.markdown(f"**Queries:** `{'` · `'.join(queries)}`")

    status_box   = st.empty()
    progress_bar = st.progress(0)
    log_box      = st.empty()
    logs         = []

    pq = queue.Queue()
    t  = threading.Thread(target=run_pipeline_with_progress, args=(cfg, pq), daemon=True)
    t.start()

    done_cfg = None
    error    = None

    while t.is_alive() or not pq.empty():
        try:
            msg_type, payload = pq.get(timeout=0.5)
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

    t.join()

    if error:
        st.error(f"Pipeline failed:\n```\n{error}\n```")
        st.stop()

    if done_cfg:
        st.success("✅ KG pipeline complete!")
        st.session_state["results"] = load_results(done_cfg)
        st.session_state["run_cfg"] = done_cfg
        st.session_state["topic"]   = topic


# ── Results display ──────────────────────────────────────────
if "results" in st.session_state:
    results = st.session_state["results"]
    cfg     = st.session_state["run_cfg"]
    topic   = st.session_state.get("topic", "")

    st.divider()
    st.header(f"Results: {topic}")

    gaps = results.get("gaps", [])
    G    = results.get("graph")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total KG Gaps",     len(gaps))
    with col2: st.metric("Missing Links",      sum(1 for g in gaps if g["type"] == "missing_link"))
    with col3: st.metric("Orphan Clusters",    sum(1 for g in gaps if g["type"] == "orphan_cluster"))
    with col4: st.metric("Decaying Concepts",  sum(1 for g in gaps if g["type"] == "temporal_decay"))

    if G:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("KG Nodes",    G.number_of_nodes())
        with col2: st.metric("KG Edges",    G.number_of_edges())
        with col3: st.metric("Components",  nx.number_weakly_connected_components(G))

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Ranked Gaps",
        "🕸️ Knowledge Graph",
        "📈 Analytics",
        "⚖️ RAG Comparison",
    ])

    # ── Tab 1 ────────────────────────────────────────────────
    with tab1:
        st.subheader("Top Research Gaps — KG Method")

        type_filter = st.multiselect(
            "Filter by type",
            ["missing_link", "orphan_cluster", "temporal_decay"],
            default=["missing_link", "orphan_cluster", "temporal_decay"],
        )

        TYPE_META = {
            "missing_link":   ("🔗", "missing", "Missing Link"),
            "orphan_cluster": ("🏝️", "orphan",  "Orphan Cluster"),
            "temporal_decay": ("📉", "decay",   "Temporal Decay"),
        }

        for g in [x for x in gaps if x["type"] in type_filter][:30]:
            icon, css, label = TYPE_META.get(g["type"], ("❓", "", g["type"]))
            st.markdown(f"""
<div class="gap-card {css}">
  <strong>#{g['rank']} {icon} {label}</strong>
  &nbsp;<code>score: {g.get('composite_score', 0):.4f}</code><br>
  <span style="color:#444">{g.get('description', '')}</span>
</div>""", unsafe_allow_html=True)

        if "gaps_df" in results:
            st.download_button(
                "⬇️ Download as CSV",
                results["gaps_df"].to_csv(index=False),
                file_name=f"kg_gaps_{topic.replace(' ', '_')}.csv",
                mime="text/csv",
            )

    # ── Tab 2 ────────────────────────────────────────────────
    with tab2:
        st.subheader("Interactive Knowledge Graph")
        st.caption("Nodes = concepts · Size = centrality · Red border = gap node · Dashed red = predicted missing link")

        if results.get("graph_html"):
            graph_path = Path(cfg["paths"]["outputs"]) / "graph_viz.html"
            if graph_path.exists():
                with open(graph_path) as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=650, scrolling=True)
        else:
            st.warning("Graph visualisation not available.")
        if G:
            st.markdown("**Top 10 most connected concepts:**")
            G_simple  = nx.DiGraph(G)
            deg_cent  = nx.degree_centrality(G_simple)
            top_nodes = sorted(deg_cent.items(), key=lambda x: -x[1])[:10]
            df_nodes  = pd.DataFrame(top_nodes, columns=["Concept", "Centrality"])
            df_nodes["Type"] = df_nodes["Concept"].apply(
                lambda n: G.nodes[n].get("type", "?") if G.has_node(n) else "?"
            )
        # cast all columns to string to avoid Arrow type conflicts
            st.table(df_nodes.astype(str))

    # ── Tab 3 ────────────────────────────────────────────────
    with tab3:
        st.subheader("Gap Analytics")

        for fig_name, caption in [
            ("gap_analysis.png",   "Gap distribution and scores"),
            ("temporal_decay.png", "Temporal decay profiles"),
            ("graph_stats.png",    "Knowledge graph statistics"),
        ]:
            p = Path(cfg["paths"]["figures"]) / fig_name
            if p.exists():
                st.image(str(p), caption=caption)

        if G:
            rel_counts = {}
            for _, _, d in G.edges(data=True):
                r = d.get("relation", "?")
                rel_counts[r] = rel_counts.get(r, 0) + 1
            df_rel = pd.DataFrame(
                sorted(rel_counts.items(), key=lambda x: -x[1]),
                columns=["Relation", "Count"],
            )
            st.markdown("**Edge relation distribution:**")
            st.bar_chart(df_rel.set_index("Relation"))

    # ── Tab 4: RAG Comparison ────────────────────────────────
    with tab4:
        st.subheader("⚖️ Method Comparison: KG vs RAG Baselines")

        if not results.get("mulla_gaps"):
            st.info(
                "RAG baselines haven't been run yet.\n\n"
                "This will run **Mulla et al. RAG** and **Simple LLM** on the same "
                f"filtered corpus ({cfg['filtering']['target_corpus_size']} papers) "
                "and compare results against the KG method."
            )

            if st.button("🤖 Run RAG Baselines", type="primary"):
                rag_status   = st.empty()
                rag_progress = st.progress(0)
                rag_logs_box = st.empty()
                rag_logs     = []

                rpq = queue.Queue()
                rt  = threading.Thread(
                    target=run_rag_with_progress,
                    args=(cfg, rpq),
                    daemon=True,
                )
                rt.start()

                rag_done  = None
                rag_error = None

                while rt.is_alive() or not rpq.empty():
                    try:
                        mt, pl = rpq.get(timeout=0.5)
                        if mt == "status":
                            rag_status.markdown(f"**{pl}**")
                            rag_logs.append(pl)
                            rag_logs_box.markdown("\n".join(f"- {l}" for l in rag_logs))
                        elif mt == "progress":
                            rag_progress.progress(pl)
                        elif mt == "done":
                            rag_done = pl
                        elif mt == "error":
                            rag_error = pl
                    except queue.Empty:
                        continue

                rt.join()

                if rag_error:
                    st.error(f"RAG baseline failed:\n```\n{rag_error}\n```")
                elif rag_done:
                    st.success("✅ RAG baselines complete!")
                    st.session_state["results"] = load_results(cfg)
                    st.rerun()

        else:
            mulla_gaps  = results.get("mulla_gaps",  [])
            simple_gaps = results.get("simple_gaps", [])
            metrics     = results.get("comparison_metrics", {})

            kg_m  = metrics.get("kg",        {})
            mu_m  = metrics.get("mulla_rag",  {})
            si_m  = metrics.get("simple_llm", {})
            ov_m  = metrics.get("overlap",    {})

            # Summary table
            st.markdown("### 📊 Method Summary")
            summary_df = pd.DataFrame({
                "Metric": [
                    "Total gaps produced",
                    "Unique gaps",
                    "Traceable evidence",
                    "Reproducible",
                    "Cross-paper retrieval",
                    "Avg gap length (words)",
                ],
                "🔷 KG (Ours)": [
                    kg_m.get("total_gaps", "-"),
                    kg_m.get("unique_gaps", "-"),
                    "✅ Subgraph path",
                    "✅ Deterministic",
                    "✅ Corpus-wide",
                    kg_m.get("avg_description_len", "-"),
                ],
                "🟡 Mulla RAG": [
                    mu_m.get("total_gaps", "-"),
                    mu_m.get("unique_gaps", "-"),
                    "❌ Free text",
                    "❌ Stochastic",
                    "✅ Top-3 similar",
                    mu_m.get("avg_gap_length", "-"),
                ],
                "⚪ Simple LLM": [
                    si_m.get("total_gaps", "-"),
                    si_m.get("unique_gaps", "-"),
                    "❌ Free text",
                    "❌ Stochastic",
                    "❌ Per-paper only",
                    si_m.get("avg_gap_length", "-"),
                ],
            })
            st.table(summary_df.set_index("Metric"))

            # Overlap
            st.markdown("### 🔀 Lexical Overlap Between Methods")
            st.caption("Jaccard similarity — lower means methods find more complementary gaps")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("KG vs Mulla RAG",    f"{ov_m.get('kg_vs_mulla',  0):.3f}")
            with col2: st.metric("KG vs Simple LLM",   f"{ov_m.get('kg_vs_simple', 0):.3f}")
            with col3: st.metric("Mulla vs Simple LLM",f"{ov_m.get('mulla_vs_simple',0):.3f}")

            st.divider()

            # Side-by-side sample
            st.markdown("### 📋 Gap Samples — Side by Side")
            paper_titles = [g.get("title", f"Paper {i}") for i, g in enumerate(mulla_gaps[:20])]
            sel = st.selectbox("Select paper", range(len(paper_titles)),
                               format_func=lambda i: paper_titles[i])

            if sel < len(mulla_gaps):
                mulla_gap = mulla_gaps[sel]
                pid       = mulla_gap.get("paper_id", "")
                simple_gap = next(
                    (g for g in simple_gaps if g.get("paper_id") == pid),
                    simple_gaps[sel] if sel < len(simple_gaps) else {},
                )
                kg_sample = gaps[:3] if gaps else []

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown('<div class="method-header">🔷 KG Method (Ours)</div>',
                                unsafe_allow_html=True)
                    for g in kg_sample:
                        css = g["type"].split("_")[0]
                        st.markdown(f"""
<div class="gap-card {css}">
  <strong>{g.get('type','').replace('_',' ').title()}</strong>
  — score {g.get('composite_score',0):.3f}<br>
  <small>{g.get('description','')[:220]}</small>
</div>""", unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="method-header">🟡 Mulla et al. RAG</div>',
                                unsafe_allow_html=True)
                    for field, label in [
                        ("research_gaps",     "Research Gaps"),
                        ("remaining_gaps",    "Remaining Gaps"),
                        ("research_direction","Direction"),
                    ]:
                        val = mulla_gap.get(field, "")
                        if val:
                            st.markdown(f"""
<div class="gap-card mulla">
  <strong>{label}</strong><br>
  <small>{val[:220]}</small>
</div>""", unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="method-header">⚪ Simple LLM</div>',
                                unsafe_allow_html=True)
                    for j in range(1, 4):
                        val = simple_gap.get(f"gap_{j}", "")
                        if val:
                            st.markdown(f"""
<div class="gap-card simple">
  <strong>Gap {j}</strong><br>
  <small>{val[:220]}</small>
</div>""", unsafe_allow_html=True)

            st.divider()
            if metrics:
                st.download_button(
                    "⬇️ Download comparison metrics (JSON)",
                    json.dumps(metrics, indent=2),
                    file_name="comparison_metrics.json",
                    mime="application/json",
                )