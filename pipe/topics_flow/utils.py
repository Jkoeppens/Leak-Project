# === pipe/topics_flow/utils.py ===
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # kein GUI nötig
import matplotlib.pyplot as plt

def export_cluster_pies(pies: pd.DataFrame, out_dir: Path):
    """
    Erzeugt kleine Pie-Charts (PNG) pro Cluster.
    Erwartet Spalten: cluster_id, thread_topic_id, share.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []

    for cid, g in pies.groupby("cluster_id"):
        shares = g["share"].tolist()
        if not shares or sum(shares) == 0:
            continue

        fig, ax = plt.subplots(figsize=(1.0, 1.0))
        ax.pie(
            shares,
            labels=None,
            colors=plt.cm.tab20.colors,
            startangle=90
        )
        ax.axis("equal")
        plt.tight_layout(pad=0)
        out_path = out_dir / f"{cid}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight", transparent=True)
        plt.close(fig)
        index.append({"cluster_id": cid, "pie_path": out_path.name})

    pd.DataFrame(index).to_csv(out_dir / "cluster_pies_index.csv", index=False)
    return out_dir