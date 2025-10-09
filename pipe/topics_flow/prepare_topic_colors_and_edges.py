# === pipe/topics_flow/prepare_topic_colors_and_edges.py ===
"""
Bereitet farbcodierte Topics und Kanteninformationen für das Organisationsnetzwerk auf.
Automatische Schema-Erkennung für topic_flows (alt/neu).
"""

import pandas as pd
from pathlib import Path
from matplotlib import colormaps
import numpy as np
import time


def prepare_topic_colors_and_edges(env: dict, period: str | None = None):
    print("\n=== [prepare_topic_colors_and_edges] Start ===")
    t0 = time.time()

    org_dir = Path(env["outputs"]["org_dir"])
    topics_dir = Path(env["outputs"]["topics_dir"])

    flow_file = org_dir / "topic_flows_filtered.csv"
    pies_file = org_dir / "cluster_topic_pies_summary.csv"

    assert flow_file.exists(), f"❌ Fehlende Datei: {flow_file}"
    assert pies_file.exists(), f"❌ Fehlende Datei: {pies_file}"

    flows = pd.read_csv(flow_file)
    pies = pd.read_csv(pies_file)
    print(f"[ok] Eingabedateien gefunden. flows={len(flows):,}  pies={len(pies):,}")

    # === 🔧 SCHEMA-HARMONISIERUNG: FLOWS ===
    cols = set(flows.columns)

    if {"sender_cluster", "recipient_cluster", "topic_active"}.issubset(cols):
        flows = flows.rename(columns={
            "sender_cluster": "source",
            "recipient_cluster": "target",
            "topic_active": "topic_id"
        })
        print("[ok] Neues Schema erkannt (sender_cluster / recipient_cluster / topic_active)")

    elif {"src", "dst", "thread_topic_id"}.issubset(cols):
        flows = flows.rename(columns={
            "src": "source",
            "dst": "target",
            "thread_topic_id": "topic_id"
        })
        print("[ok] Altes Schema erkannt (src / dst / thread_topic_id)")

    required = {"source", "target", "topic_id", "weight"}
    missing = [c for c in required if c not in flows.columns]
    if missing:
        raise ValueError(f"❌ Fehlende Spalten in flows: {missing}")

    # === 🔧 SCHEMA-HARMONISIERUNG: PIES ===
    if "thread_topic_id" in pies.columns:
        pies = pies.rename(columns={"thread_topic_id": "topic_id"})
        print("[ok] Pies: Spalte 'thread_topic_id' → 'topic_id' umbenannt")

    elif "topic_active" in pies.columns:
        pies = pies.rename(columns={"topic_active": "topic_id"})
        print("[ok] Pies: Spalte 'topic_active' → 'topic_id' umbenannt")

    # === 🎨 Farben vorbereiten ===
    topic_ids = sorted(pies["topic_id"].dropna().unique())
    cmap = colormaps.get_cmap("tab20")
    color_map = {tid: cmap(i % 20) for i, tid in enumerate(topic_ids)}

    # Farben als Hex konvertieren
    color_hex = {
        tid: "#{:02x}{:02x}{:02x}".format(
            int(255 * color_map[tid][0]),
            int(255 * color_map[tid][1]),
            int(255 * color_map[tid][2])
        )
        for tid in color_map
    }

    flows["color"] = flows["topic_id"].map(color_hex).fillna("#999999")

    # === 📈 Linienbreite skalieren ===
    flows["width"] = np.log1p(flows["weight"]) / np.log1p(flows["weight"]).max() * 10

    # === 🧩 Edges / Nodes speichern ===
    viz_dir = org_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    edges_out = viz_dir / "edges_topics.csv"
    nodes_out = viz_dir / "nodes_topics.csv"
    colors_out = viz_dir / "topic_colors.csv"

    flows["edge_id"] = flows.apply(
        lambda r: f"{r['source']}||{r['target']}||{r['topic_id']}", axis=1
    )
    flows.to_csv(edges_out, index=False)
    print(f"[write] {edges_out} ({len(flows):,} edges)")

    # === Nodes ===
    nodes = pd.DataFrame({
        "id": pd.unique(flows[["source", "target"]].values.ravel("K")),
        "group": "org_cluster"
    })
    nodes.to_csv(nodes_out, index=False)
    print(f"[write] {nodes_out} ({len(nodes):,} nodes)")

    # === Farben ===
    pd.DataFrame(list(color_hex.items()), columns=["topic_id", "color"]).to_csv(colors_out, index=False)
    print(f"[write] {colors_out} ({len(color_hex):,} topics)")

    print(f"\n✅ [prepare_topic_colors_and_edges] abgeschlossen in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    prepare_topic_colors_and_edges(env)