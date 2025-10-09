# === pipe/topics_flow/cluster_topic_pies.py ===
"""
Erzeugt Topic-Verteilungen pro organisationalem Cluster (z. B. Abteilung/Team)
basierend auf den in levels.csv definierten Hierarchieebenen.

Unterstützt zeitliche Aggregation (periodisch oder rolling).

Input:
  - events_with_topics.csv  (aus vorherigem Schritt)
  - infomap_levels.csv      (aus Hierarchie)
Output:
  - cluster_topic_pies.csv
  - cluster_topic_pies_summary.csv (pro Zeitraum oder global)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re


def cluster_topic_pies(env):
    print("\n=== [cluster_topic_pies] Starte Verarbeitung ===")

    topics_dir = Path(env["outputs"]["topics_dir"])
    org_dir = Path(env["outputs"]["org_dir"])
    levels_csv = Path(env["paths"]["levels_csv"])

    events_path = topics_dir / "events_with_topics.csv"
    assert events_path.exists(), f"❌ fehlt: {events_path}"
    assert levels_csv.exists(), f"❌ fehlt: {levels_csv}"

    events = pd.read_csv(events_path, low_memory=False)
    levels = pd.read_csv(levels_csv, low_memory=False)
    print(f"[load] Events: {len(events)} Zeilen | Levels: {len(levels)} Zeilen")

    # --- Robustheit: Spalten prüfen ---
    required_cols = {"cluster_id", "thread_topic_id", "weight"}
    missing = required_cols - set(events.columns)
    if missing:
        # Fallback: schätze Gewicht über event_count oder Wahrscheinlichkeit
        if "probability" in events.columns:
            events["weight"] = events["probability"]
        elif "event_id" in events.columns:
            events["weight"] = 1.0
        else:
            raise ValueError(f"❌ Fehlende Spalten in events_with_topics.csv: {missing}")

    # --- Cluster-ID normalisieren ---
    def normalize_cluster(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        if re.match(r"^\d+(\.\d+)?$", s):
            s = str(int(float(s)))
        return s

    events["cluster_id"] = events["cluster_id"].apply(normalize_cluster)
    events = events.dropna(subset=["cluster_id", "thread_topic_id"])

    # --- Zeitdimension vorbereiten ---
    if "date" in events.columns:
        events["date"] = pd.to_datetime(events["date"], errors="coerce")
        events = events.dropna(subset=["date"])
        events["period"] = events["date"].dt.to_period(env["runtime"].get("time_freq", "M"))
    else:
        events["period"] = "ALL"

    time_mode = env["runtime"].get("time_mode", "off")
    group_cols = (
        ["period", "cluster_id", "thread_topic_id"]
        if time_mode != "off"
        else ["cluster_id", "thread_topic_id"]
    )

    # --- Aggregation ---
    pies = (
        events.groupby(group_cols)
        .agg(
            weight=("weight", "sum"),
            n_events=("thread_topic_id", "count"),
            avg_prob=("weight", "mean")
        )
        .reset_index()
    )

    # --- Anteil & Normalisierung ---
    pies["weight_total"] = pies.groupby(
        group_cols[:-1]
    )["weight"].transform("sum")
    pies["share"] = pies["weight"] / pies["weight_total"].replace(0, np.nan)

    # --- Label-Spalte sicherstellen ---
    if "topic_label" not in events.columns:
        pies["topic_label"] = "topic_" + pies["thread_topic_id"].astype(str)
    else:
        pies = pies.merge(
            events[["thread_topic_id", "topic_label"]].drop_duplicates(),
            on="thread_topic_id",
            how="left",
        )

    # --- Zusammenfassung (pro Cluster: Top 3 Topics) ---
    def summarize_group(g):
        g = g.sort_values("weight", ascending=False).head(3)
        labs = " / ".join(f"{t} ({round(100*w/sum(g['weight']),1)}%)"
                          for t, w in zip(g["topic_label"], g["weight"]))
        return pd.Series({
            "n_topics": len(g),
            "top_topics": labs,
            "total_weight": g["weight"].sum()
        })

    pies_summary = (
        pies.groupby(group_cols[:-1])
        .apply(summarize_group)
        .reset_index()
    )

    # --- Export ---
    org_dir.mkdir(parents=True, exist_ok=True)

    def make_output(name, period=None):
        if period == "ALL" or time_mode == "off":
            return org_dir / f"{name}.csv"
        else:
            return org_dir / f"{name}_{period}.csv"

    if time_mode == "off":
        pies.to_csv(make_output("cluster_topic_pies"), index=False)
        pies_summary.to_csv(make_output("cluster_topic_pies_summary"), index=False)
    else:
        for p, sub in pies.groupby("period"):
            sub.to_csv(make_output("cluster_topic_pies", p), index=False)
        for p, sub in pies_summary.groupby("period"):
            sub.to_csv(make_output("cluster_topic_pies_summary", p), index=False)
    # --- Optionaler Mini-Pie-Export (Hook) ---
    if env["runtime"].get("export_pies", False):
        try:
            from pipe.topics_flow.utils import export_cluster_pies
            viz_dir = org_dir / "viz" / "pies"
            export_cluster_pies(pies, viz_dir)
            print(f"[viz] Mini-Pies exportiert nach {viz_dir}")
        except Exception as e:
            print(f"[warn] Mini-Pie-Export übersprungen: {e}")
    print(f"[done] {len(pies_summary)} Cluster-Topic-Zusammenfassungen exportiert")
    print("✅ [cluster_topic_pies] abgeschlossen.")
    return pies, pies_summary


# -----------------------------------------------------
if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    cluster_topic_pies(env)