"""
Erzeugt Topic-Flows zwischen Clustern basierend auf E-Mail-Ereignissen.
Verbindet event_actor.csv (Absender/Empfänger) mit infomap_levels.csv (Cluster)
und events_with_thread_topics.csv (Themenzuordnung pro Event).

Ergebnis:
  reports/orgchart/topic_flows.csv
mit Spalten:
  sender_cluster, recipient_cluster, thread_topic_id, weight, n_events
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any


def topic_flows_between_clusters(env: Dict[str, Any]) -> None:
    root = Path(env["paths"]["root"])
    clean_dir = Path(env["paths"]["clean_dir"])
    org_dir = Path(env["outputs"]["org_dir"])
    levels_csv = Path(env["paths"]["levels_csv"])

    events_csv = clean_dir / "events.csv"
    actors_csv = clean_dir / "event_actor.csv"
    topics_csv = org_dir / "events_with_thread_topics.csv"
    out_csv = org_dir / "topic_flows.csv"

    print("=== [topic_flows_between_clusters] ===")

    # --- 1️⃣ Lade Daten ---
    ev = pd.read_csv(events_csv)
    ea = pd.read_csv(actors_csv)
    lv = pd.read_csv(levels_csv)
    et = pd.read_csv(topics_csv)

    print(f"[load] events={len(ev):,} | actors={len(ea):,} | levels={len(lv):,} | topics={len(et):,}")
    print(f"[cols] levels: {list(lv.columns)}")

    # --- 2️⃣ Baue Kommunikations-Paare (from → to) ---
    senders = ea[ea["role_in_event"] == "from"].rename(columns={"email": "email_sender"})
    recipients = ea[ea["role_in_event"] == "to"].rename(columns={"email": "email_recipient"})

    pairs = pd.merge(senders, recipients, on="event_id")
    print(f"[pairs] gebildet: {len(pairs):,}")
    print(pairs.head(3))

    # --- 3️⃣ Verbinde mit Cluster-Informationen ---
    lv["node"] = lv["node"].str.strip().str.lower()
    pairs["email_sender"] = pairs["email_sender"].str.strip().str.lower()
    pairs["email_recipient"] = pairs["email_recipient"].str.strip().str.lower()

    cluster_map = lv.set_index("node")["module_path"].to_dict()
    pairs["cluster_sender"] = pairs["email_sender"].map(cluster_map)
    pairs["cluster_recipient"] = pairs["email_recipient"].map(cluster_map)

    matched = pairs["cluster_sender"].notna().mean() * 100
    print(f"[map] Cluster gemappt: {matched:.1f}% der Sender, {pairs['cluster_recipient'].notna().mean()*100:.1f}% der Empfänger")
    print(pairs.head(5)[["email_sender", "email_recipient", "cluster_sender", "cluster_recipient"]])

    # --- 4️⃣ Verbinde Topics ---
    if "thread_topic_id" not in et.columns:
        raise ValueError("❌ 'events_with_thread_topics.csv' muss 'thread_topic_id' enthalten.")

    pairs = pairs.merge(et[["event_id", "thread_topic_id"]], on="event_id", how="left")

    # --- 5️⃣ Entferne unzugeordnete Topics (-1 oder NaN) ---
    print(f"[filter] Vorher: {len(pairs):,} Zeilen")
    pairs = pairs[pairs["thread_topic_id"].notna()]
    pairs = pairs[pairs["thread_topic_id"] != -1]
    print(f"[filter] Nachher: {len(pairs):,} Zeilen | Topics={pairs['thread_topic_id'].nunique()}")

    # --- 6️⃣ Aggregiere zu Flows ---
    flows = (
        pairs.groupby(["cluster_sender", "cluster_recipient", "thread_topic_id"], as_index=False)
        .agg(weight=("event_id", "count"), n_events=("event_id", "nunique"))
    )

    # --- 7️⃣ Entferne Self-Loops ---
    self_loops = flows["cluster_sender"] == flows["cluster_recipient"]
    num_self = self_loops.sum()
    flows = flows[~self_loops]
    print(f"[flows] ohne Self-Loops: {len(flows):,} (entfernt {num_self})")

    # --- 8️⃣ Ausgabe ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    flows.to_csv(out_csv, index=False)
    print(f"[save] {out_csv} ✅ rows={len(flows):,}")
    print(flows.head(10))


# === Entry Point (wenn direkt ausgeführt) ===
if __name__ == "__main__":
    import sys
    sys.path.append("/content/Leak-Project")
    from pipe.topics_flow.setup_env import setup_environment

    env = setup_environment()
    topic_flows_between_clusters(env)