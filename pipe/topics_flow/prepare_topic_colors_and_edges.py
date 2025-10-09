# === pipe/topics_flow/prepare_topic_colors_and_edges.py ===
"""
Bereitet Farbcodes und Netzwerk-Edges für die Orgchart-Visualisierung vor.

Input:
  - topic_flows_filtered.csv (oder periodische Varianten)
  - cluster_topic_pies_summary.csv
  - topic_info_labels.csv (optional, für Farbzuordnung)

Output:
  - reports/orgchart/viz/nodes_topics.csv
  - reports/orgchart/viz/edges_topics.csv
  - reports/orgchart/viz/topic_colors.csv
"""

import pandas as pd
from pathlib import Path
from matplotlib import cm
import numpy as np
import time


def prepare_topic_colors_and_edges(env: dict, period: str | None = None):
    print("\n=== [prepare_topic_colors_and_edges] Start ===")
    t0 = time.time()

    org_dir = Path(env["outputs"]["org_dir"])
    topics_dir = Path(env["outputs"]["topics_dir"])

    # --------------------------------------------------
    # Eingabedateien prüfen
    # --------------------------------------------------
    flow_file = (
        org_dir / f"topic_flows_filtered_{period.replace('-', '').replace('_', '')}.csv"
        if period
        else org_dir / "topic_flows_filtered.csv"
    )
    pies_file = org_dir / "cluster_topic_pies_summary.csv"
    labels_file = topics_dir / "topic_info_labels.csv"

    for f in [flow_file, pies_file]:
        assert f.exists(), f"❌ Datei fehlt: {f}"
    if not labels_file.exists():
        print(f"⚠️ Keine topic_info_labels.csv gefunden → neutrale Farben")

    # --------------------------------------------------
    # CSVs laden
    # --------------------------------------------------
    flows = pd.read_csv(flow_file)
    pies = pd.read_csv(pies_file)
    labels = pd.read_csv(labels_file) if labels_file.exists() else pd.DataFrame()

    print(f"[ok] Eingelesen: flows={len(flows):,}, pies={len(pies):,}")

    # --------------------------------------------------
    # Schema validieren / anpassen
    # --------------------------------------------------
    if "topic_active" not in flows.columns:
        if "thread_topic_id" in flows.columns:
            flows = flows.rename(columns={"thread_topic_id": "topic_active"})
        else:
            raise ValueError("❌ Keine topic_active/thread_topic_id-Spalte in flows gefunden")

    if "topic_name_active" not in flows.columns and "topic_label" in flows.columns:
        flows = flows.rename(columns={"topic_label": "topic_name_active"})

    if "topic_name_active" not in flows.columns:
        flows["topic_name_active"] = "unknown"

    # --------------------------------------------------
    # Topics für Farben bestimmen
    # --------------------------------------------------
    topic_ids = sorted(flows["topic_active"].dropna().unique().tolist())
    cmap = cm.get_cmap("tab20", max(10, len(topic_ids)))

    colors = {
        tid: cm.colors.to_hex(cmap(i / len(topic_ids))) for i, tid in enumerate(topic_ids)
    }
    flows["color"] = flows["topic_active"].map(colors)
    flows["width"] = flows["weight"] / flows["weight"].max() * 8

    # --------------------------------------------------
    # Nodes generieren
    # --------------------------------------------------
    nodes_sender = flows["sender_cluster"].unique().tolist()
    nodes_recipient = flows["recipient_cluster"].unique().tolist()
    all_nodes = sorted(set(nodes_sender + nodes_recipient))

    nodes_df = pd.DataFrame({"node_id": all_nodes})
    nodes_df["n_out"] = nodes_df["node_id"].map(flows["sender_cluster"].value_counts())
    nodes_df["n_in"] = nodes_df["node_id"].map(flows["recipient_cluster"].value_counts())
    nodes_df = nodes_df.fillna(0)
    nodes_df["degree"] = nodes_df["n_out"] + nodes_df["n_in"]

    # --------------------------------------------------
    # Outputs sichern
    # --------------------------------------------------
    viz_dir = org_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    nodes_out = viz_dir / "nodes_topics.csv"
    edges_out = viz_dir / "edges_topics.csv"
    color_out = viz_dir / "topic_colors.csv"

    nodes_df.to_csv(nodes_out, index=False)
    flows.to_csv(edges_out, index=False)

    pd.DataFrame(
        [{"topic_active": k, "color": v} for k, v in colors.items()]
    ).to_csv(color_out, index=False)

    print(f"[write] {nodes_out.name}  ({len(nodes_df):,} nodes)")
    print(f"[write] {edges_out.name}  ({len(flows):,} edges)")
    print(f"[write] {color_out.name}  ({len(colors):,} topics)")
    print(f"[done] Dauer: {time.time() - t0:.1f}s")
    print("✅ [prepare_topic_colors_and_edges] abgeschlossen.")


# --------------------------------------------------
# CLI / Run Support
# --------------------------------------------------
if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    prepare_topic_colors_and_edges(env)