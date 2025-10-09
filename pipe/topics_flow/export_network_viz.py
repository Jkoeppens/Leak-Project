# === pipe/topics_flow/export_network_viz.py ===
"""
Exportiert ein interaktives vis-network (pyvis) HTML:
- erkennt automatisch Spaltennamen in nodes_topics.csv / edges_topics.csv
- nutzt hierarchische Struktur (":"-getrennte IDs)
- ignoriert ungültige Edges oder fehlende Nodes
- unterstützt Perioden-Suffixe (z. B. topic_flows_filtered_2001Q1.csv)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pyvis.network import Network
import datetime as dt

def export_network_viz(env,
                       topic_filter=None,
                       min_edge_weight=None,
                       hide_self_loops=True,
                       bg_color="#ffffff",
                       max_width=8.0,
                       width_quantile=0.9,
                       height_px=900,
                       hierarchical_direction="UD"):
    print("\n=== [export_network_viz] Start ===")

    org_dir = Path(env["outputs"]["org_dir"])
    viz_dir = org_dir / "viz"
    viz_dir.mkdir(exist_ok=True, parents=True)

    nodes_path = viz_dir / "nodes_topics.csv"
    edges_path = viz_dir / "edges_topics.csv"
    assert nodes_path.exists() and edges_path.exists(), "Fehlende Eingabedateien"
    print(f"[ok] Eingabedateien: {nodes_path.name}, {edges_path.name}")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    print(f"[load] nodes={len(nodes)} | edges={len(edges)}")

    # === Schema-Erkennung ===
    node_id_col = "node_id" if "node_id" in nodes.columns else (
        "cluster_id" if "cluster_id" in nodes.columns else nodes.columns[0]
    )
    src_col = "source" if "source" in edges.columns else (
        "sender_cluster" if "sender_cluster" in edges.columns else edges.columns[0]
    )
    dst_col = "target" if "target" in edges.columns else (
        "recipient_cluster" if "recipient_cluster" in edges.columns else edges.columns[1]
    )
    weight_col = "weight" if "weight" in edges.columns else (
        "w" if "w" in edges.columns else None
    )

    print(f"[info] node_id_col={node_id_col} | src_col={src_col} | dst_col={dst_col} | weight_col={weight_col}")

    # === Filtern & vorbereiten ===
    if weight_col and min_edge_weight:
        edges = edges[edges[weight_col] >= min_edge_weight]
        print(f"[filter] edges >= {min_edge_weight}: {len(edges)}")

    if hide_self_loops:
        edges = edges[edges[src_col] != edges[dst_col]]

    edges[src_col] = edges[src_col].astype(str)
    edges[dst_col] = edges[dst_col].astype(str)
    nodes[node_id_col] = nodes[node_id_col].astype(str)

    # === Netzwerk erstellen ===
    net = Network(height=f"{height_px}px", width="100%", bgcolor=bg_color, directed=True)
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=120)

    for _, n in nodes.iterrows():
        nid = n[node_id_col]
        color = n.get("color", "#999999")
        label = str(n.get("topic_name", nid))
        size = np.sqrt(n.get("n_events", 5)) * 3
        net.add_node(nid, label=label, color=color, size=size)

    added = 0
    for _, e in edges.iterrows():
        src, dst = e[src_col], e[dst_col]
        if src not in nodes[node_id_col].values or dst not in nodes[node_id_col].values:
            continue
        width = float(e.get("width", 1.0))
        color = e.get("color", "#cccccc")
        net.add_edge(src, dst, color=color, width=width)
        added += 1

    print(f"[done] Added {added}/{len(edges)} edges")

    # === Speichern ===
    period = None
    for f in viz_dir.glob("edges_topics_*.csv"):
        p = f.stem.replace("edges_topics_", "")
        if len(p) > 0:
            period = p
    suffix = f"_{period}" if period else ""
    out_html = viz_dir / f"org_topics_hier{suffix}.html"
    net.save_graph(str(out_html))

    print(f"[write] {out_html}  | nodes={len(nodes)} edges={added}")
    print("✅ [export_network_viz] fertig.")