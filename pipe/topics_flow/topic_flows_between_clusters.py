"""
pipe.topic_flows_between_clusters
=================================
Berechnet thematische Kommunikationsflüsse zwischen organisationalen Clustern
auf Basis der Topic-Zuordnung und dem organisationalen Mapping (levels.csv).

Input:
  - events_clean.csv
  - events_with_thread_topics.csv
  - infomap_levels.csv

Output (reports/orgchart/):
  - topic_flows.csv
    -> Kanten: sender_cluster → recipient_cluster × Topic, gewichtet nach Textlänge

Parameter:
  MIN_EDGE_WEIGHT = Mindestzeichenzahl pro Verbindung
  MIN_EDGE_EVENTS = minimale Eventanzahl pro Verbindung
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


def topic_flows_between_clusters(env: dict) -> None:
    # ---- Parameter ----
    MIN_EDGE_WEIGHT = 5000
    MIN_EDGE_EVENTS = 3

    # ---- Paths ----
    clean_dir = Path(env["paths"]["clean_dir"])
    org_dir   = Path(env["outputs"]["org_dir"])
    levels_csv = Path(env["paths"]["levels_csv"])

    CLEAN_ALL          = clean_dir / "events_clean.csv"
    EVENTS_THREAD_TOPS = org_dir / "events_with_thread_topics.csv"
    LEVELS_CSV_PATH    = levels_csv
    OUT_TOPIC_FLOWS    = org_dir / "topic_flows.csv"
    OUT_TOPIC_FLOWS.parent.mkdir(parents=True, exist_ok=True)

    for p in [CLEAN_ALL, EVENTS_THREAD_TOPS, LEVELS_CSV_PATH]:
        assert p.exists(), f"Datei fehlt: {p}"

    # ---- Load base data ----
    ev = pd.read_csv(CLEAN_ALL)
    et = pd.read_csv(EVENTS_THREAD_TOPS)
    levels = pd.read_csv(LEVELS_CSV_PATH)

    # unify keys
    eid_ev = next((c for c in ev.columns if c.lower() in {"event_id","id","eid"}), None); assert eid_ev
    ev = ev.rename(columns={eid_ev:"event_id"})
    eid_et = next((c for c in et.columns if c.lower() in {"event_id","id","eid"}), None); assert eid_et
    et = et.rename(columns={eid_et:"event_id"})

    # merge topics on events
    if "thread_topic_id" not in et.columns:
        raise ValueError("events_with_thread_topics.csv muss 'thread_topic_id' enthalten.")
    ev = ev.merge(et[["event_id","thread_topic_id"]], on="event_id", how="left")

    # text length fallback
    if "char_len" not in ev.columns:
        ev["char_len"] = ev.get("text_clean","").astype(str).str.len()

    # ---- Email Normalisierung ----
    EMAIL_RE = re.compile(r'<?([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})>?', re.I)
    def norm_email(x):
        if pd.isna(x): return None
        s = str(x).strip().lower()
        m = EMAIL_RE.search(s)
        return m.group(1) if m else None

    ev["from_email_norm"] = ev.get("from_email", np.nan).map(norm_email)

    # Empfängerlisten (beliebige Trennzeichen)
    def split_emails(value):
        if pd.isna(value): return []
        parts = re.split(r"[;, ]+", str(value))
        return [norm_email(x) for x in parts if norm_email(x)]

    ev["to_emails_norm"] = ev.get("to_emails", "").apply(split_emails)

    # ---- Cluster-Mapping über levels.csv ----
    level_key = None
    for cand in ["email","mail","node","address","user","person"]:
        mask = levels.columns.str.lower() == cand
        if mask.any():
            level_key = levels.columns[mask][0]; break
    if level_key is None: level_key = levels.columns[0]

    levels["_key_email"] = levels[level_key].map(norm_email)
    CLUSTER_COL = "module_path" if "module_path" in levels.columns else levels.columns[1]

    level_map = (levels.dropna(subset=["_key_email"])
                       .drop_duplicates("_key_email")
                       .set_index("_key_email")[CLUSTER_COL])

    ev["sender_cluster"] = ev["from_email_norm"].map(level_map)
    ev = ev[ev["sender_cluster"].notna()]

    # ---- Empfänger expandieren → Zeilen duplizieren ----
    rows = []
    for _, row in ev.iterrows():
        for rcpt in row["to_emails_norm"]:
            rcpt_cluster = level_map.get(rcpt)
            if rcpt_cluster:
                rows.append({
                    "sender_cluster": row["sender_cluster"],
                    "recipient_cluster": rcpt_cluster,
                    "thread_topic_id": row["thread_topic_id"],
                    "weight": row["char_len"],
                    "event_id": row["event_id"],
                })
    flows = pd.DataFrame(rows)

    if flows.empty:
        print("[warn] Keine gültigen Kanten erzeugt – evtl. keine gemappten Empfänger.")
        return

    # ---- Aggregieren ----
    agg = flows.groupby(
        ["sender_cluster","recipient_cluster","thread_topic_id"]
    ).agg(
        weight=("weight","sum"),
        n_events=("event_id","nunique")
    ).reset_index()

    # ---- Filter ----
    agg = agg[(agg["weight"] >= MIN_EDGE_WEIGHT) & (agg["n_events"] >= MIN_EDGE_EVENTS)]

    # ---- Save ----
    agg.to_csv(OUT_TOPIC_FLOWS, index=False)
    print(f"[write] {OUT_TOPIC_FLOWS}  rows={len(agg)}")
    print(agg.head(10).to_string(index=False))


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    topic_flows_between_clusters(env)