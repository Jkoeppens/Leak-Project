# === pipe/topics_flow/topic_flows.py ===
"""
Erzeugt gerichtete Topic-Flows zwischen organisationalen Clustern
basierend auf Ereignissen (z. B. E-Mail-Threads oder Meetings).

Input:
  - cluster_topic_pies_summary.csv
  - topic_distribution_by_level.csv
  - events_with_topics.csv (optional)
  - infomap_levels.csv (zur Hierarchieprüfung)
  - topic_info_labels.csv (mit Spalten: Topic, Name, topic_label)

Output:
  - topic_flows_filtered.csv
  - optional mit Perioden-Suffix (z. B. topic_flows_filtered_2001Q1.csv)
"""

import pandas as pd
from pathlib import Path
import os
import time


# --------------------------------------------------
# Helper: Lade Topic-Label-Tabelle (robust)
# --------------------------------------------------
def load_label_table(env):
    topics_dir = Path(env["outputs"]["topics_dir"])
    label_path = topics_dir / "topic_info_labels.csv"
    assert label_path.exists(), f"❌ Datei fehlt: {label_path}"

    df = pd.read_csv(label_path)
    cols = set(df.columns)
    if {"Topic", "Name"}.issubset(cols):
        df = df.rename(columns={"Topic": "topic_active", "Name": "topic_name_active"})
    elif {"topic_active", "topic_name_active"}.issubset(cols):
        pass  # schon richtig
    elif {"Topic", "topic_label"}.issubset(cols):
        df = df.rename(columns={"Topic": "topic_active", "topic_label": "topic_name_active"})
    else:
        raise ValueError(f"⚠️ Unbekanntes Schema für topic_info_labels.csv: {list(df.columns)}")

    return df[["topic_active", "topic_name_active"]].drop_duplicates()


# --------------------------------------------------
# Hauptfunktion
# --------------------------------------------------
def topic_flows(env: dict, period: str | None = None) -> None:
    print("\n=== [topic_flows] Start ===")
    t0 = time.time()

    org_dir = Path(env["outputs"]["org_dir"])
    topics_dir = Path(env["outputs"]["topics_dir"])
    clean_dir = Path(env["paths"]["clean_dir"])

    pies_summary_path = org_dir / "cluster_topic_pies_summary.csv"
    topic_dist_path = org_dir / "topic_distribution_by_level.csv"
    levels_path = Path(env["paths"]["levels_csv"])
    events_with_topics_path = topics_dir / "events_with_topics.csv"

    # --------------------------------------------------
    # Eingabeprüfung
    # --------------------------------------------------
    for p in [pies_summary_path, topic_dist_path, levels_path]:
        assert p.exists(), f"❌ Datei fehlt: {p}"
    if not events_with_topics_path.exists():
        print(f"⚠️ Hinweis: events_with_topics.csv nicht gefunden → wird übersprungen")

    print("[ok] Eingabedateien gefunden")

    # --------------------------------------------------
    # Daten laden
    # --------------------------------------------------
    pies = pd.read_csv(pies_summary_path)
    levels = pd.read_csv(levels_path)
    label_table = load_label_table(env)

    # Optional Events (falls vorhanden)
    ev_topics = (
        pd.read_csv(events_with_topics_path)
        if events_with_topics_path.exists()
        else pd.DataFrame(columns=["event_id", "thread_topic_id", "sender_cluster", "recipient_cluster"])
    )

    print(f"[load] pies={len(pies):,} | levels={len(levels):,} | events={len(ev_topics):,}")

    # --------------------------------------------------
    # Falls period angegeben, filtere Events
    # --------------------------------------------------
    if period and "timestamp" in ev_topics.columns:
        ev_topics["timestamp"] = pd.to_datetime(ev_topics["timestamp"], errors="coerce")
        mask = ev_topics["timestamp"].dt.to_period(period.split('-')[0])  # z. B. "2001Q1"
        ev_topics = ev_topics[mask.astype(str) == period]
        print(f"[filter] Zeitraum {period}: {len(ev_topics):,} Events")

    # --------------------------------------------------
    # Cluster-Zuordnung (vereinfachte Struktur)
    # --------------------------------------------------
    pies = pies.rename(columns={"cluster_id": "sender_cluster", "thread_topic_id": "thread_topic_id"})
    pies["recipient_cluster"] = pies["sender_cluster"]
    pies["weight"] = pies["weight"].fillna(0)

    # --------------------------------------------------
    # Flows bilden
    # --------------------------------------------------
    flows = pies.groupby(["sender_cluster", "recipient_cluster", "thread_topic_id"], as_index=False)["weight"].sum()
    flows["n_events"] = flows["weight"].round().astype(int)
    flows = flows.rename(columns={"thread_topic_id": "topic_active"})
    print(f"[flows] {len(flows):,} Kanten")

    # --------------------------------------------------
    # Labels joinen
    # --------------------------------------------------
    flows = flows.merge(label_table, on="topic_active", how="left")
    missing_labels = flows["topic_name_active"].isna().sum()
    if missing_labels > 0:
        print(f"⚠️ {missing_labels} Topics ohne Label – werden ignoriert")

    # --------------------------------------------------
    # Metadaten / Visual-Daten
    # --------------------------------------------------
    flows["color"] = "#1f77b4"
    flows["width"] = flows["weight"] / flows["weight"].max() * 8
    flows["edge_id"] = (
        flows["sender_cluster"].astype(str)
        + "||"
        + flows["recipient_cluster"].astype(str)
        + "||"
        + flows["topic_active"].astype(str)
    )

    # --------------------------------------------------
    # Output schreiben
    # --------------------------------------------------
    out_path = org_dir / "topic_flows_filtered.csv"
    if period:
        out_path = org_dir / f"topic_flows_filtered_{period.replace('-', '').replace('_', '')}.csv"

    flows.to_csv(out_path, index=False)
    print(f"[save] {out_path.name}  |  rows={len(flows):,}")

    print(f"[done] Dauer: {time.time()-t0:.1f}s")
    print("✅ [topic_flows] abgeschlossen.")
    return flows


# --------------------------------------------------
# CLI / Run Support
# --------------------------------------------------
if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    topic_flows(env)