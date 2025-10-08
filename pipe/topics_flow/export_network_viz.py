"""
pipe.export_network_viz
=======================
Exportiert Netzwerkdaten (Cluster-Nodes mit Topic-Pies, farbige Topic-Kanten)
für Visualisierungstools wie Gephi, Cytoscape oder D3.js.

Erzeugt:
  - reports/orgchart/viz/nodes_topics.csv
  - reports/orgchart/viz/edges_topics.csv
  - reports/orgchart/viz/topic_colors.csv
  - (Optionale Pie-Images für spätere Verwendung)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps as mpl_cmaps


def export_network_viz(env: dict) -> None:
    org_dir    = Path(env["outputs"]["org_dir"])
    topics_dir = Path(env["outputs"]["topics_dir"])

    PIES_CSV   = org_dir / "cluster_topic_pies.csv"
    FLOWS_CSV  = org_dir / "topic_flows_filtered.csv"
    TOPIC_INFO = topics_dir / "topic_info_labels.csv"

    # ---- Outputs ----
    VIZ_DIR   = org_dir / "viz"
    NODES_CSV = VIZ_DIR / "nodes_topics.csv"
    EDGES_CSV = VIZ_DIR / "edges_topics.csv"
    TOPCOLORS = VIZ_DIR / "topic_colors.csv"
    PIE_DIR   = VIZ_DIR / "pies"

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    PIE_DIR.mkdir(parents=True, exist_ok=True)

    assert PIES_CSV.exists() and FLOWS_CSV.exists(), "Bitte 5a & 6a zuerst ausführen."

    pies  = pd.read_csv(PIES_CSV)
    flows = pd.read_csv(FLOWS_CSV)
    labels = pd.read_csv(TOPIC_INFO) if TOPIC_INFO.exists() else pd.DataFrame()

    # ---- Topic-Farben ----
    topics_all = sorted(set(pies["thread_topic_id"].dropna().unique()) |
                        set(flows["thread_topic_id"].dropna().unique()))

    cmap = mpl_cmaps.get_cmap("tab20")
    def to_hex(rgba):
        r, g, b, a = rgba
        return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
    topic_colors = {t: to_hex(cmap(i % cmap.N)) for i, t in enumerate(topics_all)}

    # optional Labels mergen
    lab = labels.copy()
    if not lab.empty:
        if "Topic" in lab.columns:
            lab = lab.rename(columns={"Topic":"thread_topic_id"})
        elif "topic_id" in lab.columns:
            lab = lab.rename(columns={"topic_id":"thread_topic_id"})
        lab = lab[["thread_topic_id"] + [c for c in lab.columns if c!="thread_topic_id"]]

    pd.DataFrame({"topic_id": list(topic_colors.keys()),
                  "color_hex": list(topic_colors.values())}) \
        .merge(lab, left_on="topic_id", right_on="thread_topic_id", how="left") \
        .drop(columns=["thread_topic_id"], errors="ignore") \
        .to_csv(TOPCOLORS, index=False)

    # ---- Nodes (Pies) ----
    wide_share  = pies.pivot_table(index="cluster_id", columns="thread_topic_id",
                                   values="share",  aggfunc="sum", fill_value=0.0)
    wide_weight = pies.pivot_table(index="cluster_id", columns="thread_topic_id",
                                   values="weight", aggfunc="sum", fill_value=0.0)

    node_weight_total = wide_weight.sum(axis=1).rename("node_weight_total")

    def make_pie_json(row):
        d = {str(int(col)): float(val) for col, val in row.items() if float(val) > 0}
        return json.dumps(d, separators=(",",":"))

    pie_json = wide_share.apply(make_pie_json, axis=1).rename("pie_json")

    nodes = pd.DataFrame(index=wide_share.index)
    nodes["node_weight_total"] = node_weight_total
    nodes["n_topics_nonzero"]  = (wide_share > 0).sum(axis=1)

    def top_topics_str(row, k=5):
        items = [(t, row[t]) for t in row.index if row[t] > 0]
        items.sort(key=lambda x: x[1], reverse=True)
        return ";".join([f"{int(t)}:{v:.2f}" for t, v in items[:k]])

    nodes["top_topics"] = wide_share.apply(top_topics_str, axis=1)

    wide_share_ren = wide_share.copy()
    wide_share_ren.columns  = [f"topic_{int(c)}_share" for c in wide_share_ren.columns]

    wide_weight_topics = wide_weight.copy()
    wide_weight_topics.columns = [f"topic_{int(c)}_weight" for c in wide_weight_topics.columns]

    nodes = nodes.join(wide_share_ren, how="left") \
                 .join(wide_weight_topics, how="left") \
                 .join(pie_json, how="left")

    nodes = nodes.reset_index().rename(columns={"cluster_id":"id"})
    nodes.to_csv(NODES_CSV, index=False)

    # ---- Edges ----
    edges = flows.copy()
    edges = edges.rename(columns={
        "sender_cluster":"source",
        "recipient_cluster":"target",
        "thread_topic_id":"topic_id"
    })
    edges["color"] = edges["topic_id"].map(topic_colors)

    q_cap = edges["weight"].quantile(0.90) if len(edges) else 1.0
    q_cap = q_cap if q_cap > 0 else 1.0
    edges["width"] = 1.0 + (np.minimum(edges["weight"], q_cap) / q_cap) * 7.0

    edges["edge_id"] = (edges["source"].astype(str) + "||" +
                        edges["target"].astype(str) + "||" +
                        edges["topic_id"].astype(str))
    edges.to_csv(EDGES_CSV, index=False)

    print(f"[write] {NODES_CSV}")
    print(f"[write] {EDGES_CSV}")
    print(f"[write] {TOPCOLORS}")


if __name__ == "__main__":
    from pipe.setup_env import setup_environment
    env = setup_environment()
    export_network_viz(env)