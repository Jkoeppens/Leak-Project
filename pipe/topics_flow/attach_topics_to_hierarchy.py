"""
pipe.attach_topics_to_hierarchy
===============================
Verknüpft Cluster-Topic-Pies mit Hierarchieebenen (infomap_levels.csv)
und ordnet Themengewichte je Level & Cluster-ID zu.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def attach_topics_to_hierarchy(env: dict):
    org_dir = Path(env["outputs"]["org_dir"])
    pies_summary = pd.read_csv(org_dir / "cluster_topic_pies_summary.csv")

    levels_csv = Path(
        env.get("inputs", {}).get("levels_csv") or env["paths"]["levels_csv"]
)

    H = pd.read_csv(levels_csv)

    # --- Helper ---
    def lv_of_cluster(cid: str) -> str:
        if not isinstance(cid, str):
            return "L?"
        return f"L{cid.count(':') + 1}"

    label_col = None
    for cand in ["topic_label", "Name", "topic_name_active"]:
        if cand in pies_summary.columns:
            label_col = cand
            break
    if label_col is None:
        label_col = "thread_topic_id"

    # --- Topics by Level ---
    topics_by_level = {}
    for _, r in pies_summary.iterrows():
        lv = lv_of_cluster(r["cluster_id"])
        cid = str(r["cluster_id"])
        if pd.isna(r.get(label_col)):
            continue
        label = (str(r[label_col]) if label_col != "thread_topic_id"
                 else f"topic{int(r['thread_topic_id'])}")
        w = float(r.get("weight", 0))
        topics_by_level.setdefault(lv, {}).setdefault(cid, []).append((label, w))

    print(f"[ok] Topics verknüpft: {sum(len(v) for v in topics_by_level.values())} Cluster mit Themen")
    print(f"Levels erkannt: {sorted(topics_by_level.keys())}")

    # --- Optionale Normalisierung ---
    # Filtere nur Cluster, die in infomap_levels existieren
    all_clusters = set(H.select_dtypes(include="object").stack().dropna().unique())
    topics_by_level = {
        lv: {cid: topics for cid, topics in d.items() if cid in all_clusters}
        for lv, d in topics_by_level.items()
    }

    print(f"[ok] Gefiltert auf bekannte Cluster in {levels_csv.name}")

    return topics_by_level


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    topics_by_level = attach_topics_to_hierarchy(env)