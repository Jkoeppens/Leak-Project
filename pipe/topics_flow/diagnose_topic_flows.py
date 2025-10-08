"""
pipe.diagnose_topic_flows
=========================
Analyse und Plausibilitäts-Check der thematischen Kommunikationsflüsse
zwischen Clustern (aus topic_flows.csv).

Features:
- Basisstatistik & Filter (Gewicht, Events, Noise)
- Top-Kanten und Top-Themen
- Detailansicht einzelner Topics
- Optionale Heatmaps / Histogramme

Output:
  reports/orgchart/topic_flows_filtered.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def diagnose_topic_flows(env: dict) -> None:
    org_dir = Path(env["outputs"]["org_dir"])
    FLOW_IN   = org_dir / "topic_flows.csv"
    FLOW_OUT  = org_dir / "topic_flows_filtered.csv"

    assert FLOW_IN.exists(), f"Fehlt: {FLOW_IN}"
    flows = pd.read_csv(FLOW_IN)

    # ---- Settings ----
    EXCLUDE_NOISE     = True      # Topic -1 entfernen
    MIN_EDGE_WEIGHT   = 5000      # Mindest-Zeichen
    MIN_EDGE_EVENTS   = 3         # Mindest-Eventanzahl
    FOCUS_TOPICS_N    = None      # z.B. 10 → nur Top-10 Topics
    SHOW_TOPIC        = None      # z.B. 30 → Detailmatrix

    # ---- Checks ----
    req = {"sender_cluster","recipient_cluster","thread_topic_id","weight","n_events"}
    missing = req - set(flows.columns)
    assert not missing, f"Spalten fehlen in topic_flows: {missing}"

    print("=== Überblick (roh) ===")
    print(f"Edges: {len(flows)} | Topics: {flows['thread_topic_id'].nunique()}")
    print("Min/Median/95%-Perzentil Gewicht:",
          int(flows["weight"].min()),
          int(flows["weight"].median()),
          int(flows["weight"].quantile(0.95)))

    # ---- Filter anwenden ----
    f = flows.copy()
    if EXCLUDE_NOISE:
        f = f[f["thread_topic_id"] != -1]
    f = f[(f["weight"] >= MIN_EDGE_WEIGHT) & (f["n_events"] >= MIN_EDGE_EVENTS)].copy()

    if FOCUS_TOPICS_N:
        topT = (f.groupby("thread_topic_id")["weight"].sum()
                  .sort_values(ascending=False).head(FOCUS_TOPICS_N).index)
        f = f[f["thread_topic_id"].isin(topT)]

    print("\n=== Überblick (gefiltert) ===")
    print(f"Edges: {len(f)} | Topics: {f['thread_topic_id'].nunique()}")
    print("Self-Loops (A→A):", int((f['sender_cluster']==f['recipient_cluster']).sum()))
    print("Cross-Cluster:", int((f['sender_cluster']!=f['recipient_cluster']).sum()))

    # ---- Top-Kanten gesamt ----
    print("\n=== Top 15 Kanten nach Gewicht (gefiltert) ===")
    print(f.sort_values("weight", ascending=False).head(15).to_string(index=False))

    # ---- Topic-Perspektive ----
    topic_tot = (f.groupby("thread_topic_id")["weight"].sum()
                   .sort_values(ascending=False).rename("topic_weight"))
    print("\n=== Top 10 Topics nach Gesamtgewicht ===")
    print(topic_tot.head(10).to_string())

    # ---- Für Top-K Topics die stärksten Kanten ----
    TOPK = 5
    print(f"\n=== Für Top-{TOPK} Topics: stärkste Kanten (jeweils Top 5) ===")
    for t in topic_tot.head(TOPK).index:
        sub = f[f["thread_topic_id"] == t].sort_values("weight", ascending=False).head(5)
        print(f"\nTopic {t}:")
        print(sub[["sender_cluster","recipient_cluster","weight","n_events"]].to_string(index=False))

    # ---- Detailansicht: einzelne Topics (optional) ----
    if SHOW_TOPIC is not None and SHOW_TOPIC in f["thread_topic_id"].unique():
        sub = f[f["thread_topic_id"] == SHOW_TOPIC]
        pivot = sub.pivot_table(index="sender_cluster", columns="recipient_cluster",
                                values="weight", aggfunc="sum", fill_value=0)
        print(f"\n=== Matrix Gewicht (Topic {SHOW_TOPIC}) – Größe: {pivot.shape} ===")
        if pivot.shape[0] <= 30 and pivot.shape[1] <= 30:
            print(pivot.to_string())

        plt.figure()
        plt.hist(sub["weight"], bins=30)
        plt.title(f"Gewichtsverteilung Topic {SHOW_TOPIC}")
        plt.xlabel("weight (Zeichen)")
        plt.ylabel("Kanten-Anzahl")
        plt.show()

    # ---- Speichern ----
    f.to_csv(FLOW_OUT, index=False)
    print(f"\n[write] {FLOW_OUT}  rows={len(f)}  "
          f"Filter: noise={'off' if not EXCLUDE_NOISE else 'on'}, "
          f"min_w={MIN_EDGE_WEIGHT}, min_n={MIN_EDGE_EVENTS}, topN={FOCUS_TOPICS_N}")


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    diagnose_topic_flows(env)