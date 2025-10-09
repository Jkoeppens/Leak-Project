# === pipe/topics_flow/topic_flows.py ===
"""
Erzeugt gerichtete Topic-Flows zwischen organisationalen Clustern
basierend auf Ereignissen (z. B. E-Mail-Threads oder Meetings).

Input:
  - cluster_topic_pies_summary.csv
  - topic_distribution_by_level.csv
  - events_with_topics.csv (optional)
  - infomap_levels.csv (zur Hierarchieprüfung)

Output:
  - topic_flows_filtered.csv (bereit für Visualisierung)
"""

import pandas as pd
from pathlib import Path

def topic_flows(env: dict, period: str | None = None) -> None:
    print("\n=== [topic_flows] Start ===")

    org_dir = Path(env["outputs"]["org_dir"])
    topics_dir = Path(env["outputs"]["topics_dir"])

    # === Eingabedateien prüfen ===
    pies_path = org_dir / "cluster_topic_pies_summary.csv"
    levels_path = Path(env["paths"]["levels_csv"])
    events_topics_path = topics_dir / "events_with_topics.csv"

    for p in [pies_path, levels_path]:
        assert p.exists(), f"❌ fehlt: {p}"

    pies = pd.read_csv(pies_path)
    levels = pd.read_csv(levels_path)

    print(f"[ok] Pies: {len(pies)}, Levels: {len(levels)}")

    # === Cluster-Spalten angleichen ===
    cluster_col = "cluster_id" if "cluster_id" in pies.columns else "module_path"
    pies[cluster_col] = pies[cluster_col].astype(str).str.strip()
    levels["module_path"] = levels["module_path"].astype(str).str.strip()

    # === Topic-Zuordnungen pro Cluster ===
    topic_weights = (
        pies.groupby([cluster_col, "thread_topic_id"])
        .agg(weight=("weight", "sum"), n_events=("n_events", "sum"))
        .reset_index()
    )

    print(f"[ok] Topic-Cluster Paare: {len(topic_weights)}")

    # === Simulierte oder echte Verbindungen ===
    # Für diese Version: Verbinde Cluster mit identischem Level-Elternteil
    # (dient als semantische Nähe zwischen Abteilungen)
    levels["parent"] = levels["module_path"].apply(lambda x: ":".join(x.split(":")[:-1]) if ":" in x else None)

    # Join: alle Cluster mit gemeinsamem parent
    edges = (
        levels.merge(levels, on="parent", suffixes=("_sender", "_recipient"))
        .loc[:, ["module_path_sender", "module_path_recipient", "parent"]]
        .rename(columns={
            "module_path_sender": "sender_cluster",
            "module_path_recipient": "recipient_cluster"
        })
        .dropna(subset=["sender_cluster", "recipient_cluster"])
        .drop_duplicates()
    )

    # === Themengewicht pro Verbindung ===
    # Aggregiere über Topics, hier beispielhaft gleiche weight übernehmen
    edges = edges.assign(thread_topic_id=None, weight=None, n_events=None)

    # Merge optional mit Topic-Infos, falls Cluster überlappt
    edges = edges.merge(
        topic_weights.rename(columns={"cluster_id": "sender_cluster"}),
        on="sender_cluster",
        how="left"
    )

    # === Cleaning ===
    edges = edges.dropna(subset=["sender_cluster", "recipient_cluster"])
    edges = edges[edges["sender_cluster"] != edges["recipient_cluster"]]  # keine self-loops

    print(f"[ok] Edges generiert: {len(edges)}")
    print("[preview]")
    print(edges.head(5))

    # === Speichern ===
    out_path = org_dir / "topic_flows_filtered.csv"
    edges.to_csv(out_path, index=False)
    print(f"[save] {out_path} ({len(edges)} Zeilen)")

    print("✅ [topic_flows] abgeschlossen.")