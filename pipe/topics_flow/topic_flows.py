# === pipe/topics_flow/topic_flows.py ===
"""
Erzeugt Flüsse (Edges) zwischen organisationalen Clustern auf Basis
von Events mit Topics. Unterstützt optional zeitliche Aggregation
(periodisch oder rolling).

Input:
  - events_with_topics.csv
  - infomap_levels.csv (zur Hierarchieprüfung)

Output:
  - topic_flows_filtered.csv (oder topic_flows_filtered_YYYY-MM.csv)

Voraussetzung:
  setup_environment() liefert env mit:
    env["paths"]["levels_csv"]
    env["outputs"]["topics_dir"]
    env["runtime"]["time_mode"] ∈ {"off","periodic","rolling"}
"""

import pandas as pd
from pathlib import Path
import numpy as np
import re

# -----------------------------------------------------
def topic_flows(env):
    print("\n=== [topic_flows] Starte Verarbeitung ===")

    topics_dir = Path(env["outputs"]["topics_dir"])
    levels_csv = Path(env["paths"]["levels_csv"])
    clean_dir = Path(env["paths"]["clean_dir"])

    # --- Eingabedateien ---
    events_path = topics_dir / "events_with_topics.csv"
    assert events_path.exists(), f"❌ fehlt: {events_path}"

    print(f"[load] Events: {events_path}")
    events = pd.read_csv(events_path, low_memory=False)

    # --- Robustheit: Spalten-Check ---
    required_cols = {"sender_cluster", "recipient_cluster", "thread_topic_id"}
    missing = required_cols - set(events.columns)
    if missing:
        raise ValueError(f"❌ Fehlende Spalten in events_with_topics.csv: {missing}")

    # --- Optional: Datum vorbereiten ---
    if "date" in events.columns:
        events["date"] = pd.to_datetime(events["date"], errors="coerce")
        events = events.dropna(subset=["date"])
        events["period"] = events["date"].dt.to_period(env["runtime"].get("time_freq", "M"))
    else:
        events["period"] = "ALL"

    # --- Filter ---
    # Nur relevante Flüsse behalten (Topic nicht NaN, Cluster definiert)
    events = events.dropna(subset=["thread_topic_id", "sender_cluster", "recipient_cluster"])
    events["thread_topic_id"] = events["thread_topic_id"].astype(int)

    # --- Format-Korrektur für Cluster-IDs (str statt float) ---
    def normalize_cluster_id(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        # Entferne Nachkommastellen, falls versehentlich numerisch
        if re.match(r"^\d+(\.\d+)?$", s):
            s = str(int(float(s)))
        return s

    events["sender_cluster"] = events["sender_cluster"].apply(normalize_cluster_id)
    events["recipient_cluster"] = events["recipient_cluster"].apply(normalize_cluster_id)

    # --- Aggregation ---
    time_mode = env["runtime"].get("time_mode", "off")
    group_cols = (
        ["period", "sender_cluster", "recipient_cluster", "thread_topic_id"]
        if time_mode != "off"
        else ["sender_cluster", "recipient_cluster", "thread_topic_id"]
    )

    print(f"[agg] Gruppiere über: {group_cols}")
    flows = (
        events.groupby(group_cols)
        .agg(
            weight=("thread_topic_id", "count"),
            n_events=("thread_topic_id", "count")
        )
        .reset_index()
    )

    # --- Filter nach Gewicht ---
    min_weight = env.get("thresholds", {}).get("min_flow_weight", 3)
    flows = flows[flows["weight"] >= min_weight]
    print(f"[filter] Flüsse >= {min_weight} behalten → {len(flows)} Zeilen")

    # --- Farben & Format vorbereiten ---
    flows["color"] = "#1f77b4"
    flows["width"] = np.sqrt(flows["weight"]) / 4.0
    flows["edge_id"] = (
        flows["sender_cluster"].astype(str)
        + "||"
        + flows["recipient_cluster"].astype(str)
        + "||"
        + flows["thread_topic_id"].astype(str)
    )

    # --- Sanity: Hierarchieprüfung ---
    if levels_csv.exists():
        H = pd.read_csv(levels_csv)
        known_clusters = set(H.iloc[:, -1].astype(str).unique())
        missing_src = set(flows["sender_cluster"]) - known_clusters
        missing_dst = set(flows["recipient_cluster"]) - known_clusters
        if missing_src or missing_dst:
            print(
                f"[warn] {len(missing_src)} unbekannte Sender, "
                f"{len(missing_dst)} unbekannte Empfänger in Hierarchie"
            )

    # --- Export ---
    topics_dir.mkdir(parents=True, exist_ok=True)

    if time_mode == "off":
        out_path = topics_dir / "topic_flows_filtered.csv"
        flows.to_csv(out_path, index=False)
        print(f"[save] {out_path} ({len(flows)} Zeilen)")
    else:
        for p, sub in flows.groupby("period"):
            out_path = topics_dir / f"topic_flows_filtered_{p}.csv"
            sub.to_csv(out_path, index=False)
        print(f"[save] {len(flows)} Zeilen in periodischen Dateien exportiert")

    print("✅ [topic_flows] abgeschlossen.")
    return flows


# -----------------------------------------------------
if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    topic_flows(env)