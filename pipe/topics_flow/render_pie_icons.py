"""
pipe.render_pie_icons
=====================
Erzeugt Pie-Chart-Icons je Cluster auf Basis der Topic-Verteilungen
(share-Spalten in nodes_topics.csv). Diese Icons können in Gephi,
D3.js oder Webmaps als Cluster-Symbole verwendet werden.

Input:
  - reports/orgchart/viz/nodes_topics.csv
  - reports/orgchart/viz/topic_colors.csv

Output:
  - reports/orgchart/viz/pies/<cluster_id>.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_pie_icons(env: dict) -> None:
    org_dir = Path(env["outputs"]["org_dir"])
    VIZ_DIR   = org_dir / "viz"
    NODES_CSV = VIZ_DIR / "nodes_topics.csv"
    TOPCOLORS = VIZ_DIR / "topic_colors.csv"
    PIE_DIR   = VIZ_DIR / "pies"
    PIE_DIR.mkdir(parents=True, exist_ok=True)

    assert NODES_CSV.exists(), f"Fehlt: {NODES_CSV}"
    assert TOPCOLORS.exists(), f"Fehlt: {TOPCOLORS}"

    nodes = pd.read_csv(NODES_CSV).set_index("id")
    tcols = pd.read_csv(TOPCOLORS)

    # Farbzuordnung
    topic_colors = dict(zip(tcols["topic_id"], tcols["color_hex"]))

    # Topic-Spalten finden
    share_cols = [c for c in nodes.columns if c.endswith("_share")]
    if not share_cols:
        raise ValueError("Keine *_share-Spalten in nodes_topics.csv gefunden.")
    topics = [int(c.split("_")[1]) for c in share_cols]

    print(f"[info] {len(nodes)} Cluster | {len(topics)} Topics mit Shares")

    # Rendern
    for cid, row in nodes[share_cols].iterrows():
        vals = row.values.astype(float)
        if vals.sum() <= 0:
            continue
        cols = [topic_colors.get(int(t), "#cccccc") for t in topics]

        fig, ax = plt.subplots(figsize=(0.9, 0.9), dpi=200)
        ax.pie(vals, colors=cols, radius=1.0, startangle=90, wedgeprops={"linewidth": 0})
        ax.set(aspect="equal"); ax.axis("off")

        safe_name = str(cid).replace(":", "_").replace("/", "_")
        out = PIE_DIR / f"{safe_name}.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print(f"[done] {PIE_DIR} — Pie-Icons für {len(nodes)} Cluster erzeugt.")


if __name__ == "__main__":
    from pipe.setup_env import setup_environment
    env = setup_environment()
    render_pie_icons(env)