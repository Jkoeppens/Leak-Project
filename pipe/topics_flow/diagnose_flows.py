"""
pipe.diagnose_flows
===================
Diagnostik und Plausibilitätscheck der erzeugten Topic-Flussdaten.
Liest die im reports/orgchart/ erzeugten CSVs ein und zeigt
Statistiken, Verteilungen und potenzielle Artefakte an.

Outputs nur auf der Konsole (kein Schreibvorgang).

Beispiel:
---------
from pipe.setup_env import setup_environment
from pipe.diagnose_flows import diagnose_flows

env = setup_environment()
diagnose_flows(env)
"""

from pathlib import Path
import pandas as pd


def diagnose_flows(env: dict) -> None:
    org_dir = Path(env["outputs"]["org_dir"])
    FLOW      = org_dir / "topic_flow_edges_by_level.csv"
    FLOW_TOT  = org_dir / "topic_flow_edges_total.csv"
    SEND_DIST = org_dir / "topic_distribution_by_sender_level.csv"
    REC_DIST  = org_dir / "topic_distribution_by_recipient_level.csv"
    EVENTS    = org_dir / "events_with_thread_topics.csv"

    for p in [FLOW, FLOW_TOT, SEND_DIST, REC_DIST, EVENTS]:
        assert p.exists(), f"Datei fehlt: {p}"

    flow       = pd.read_csv(FLOW)
    flow_total = pd.read_csv(FLOW_TOT)
    send_dist  = pd.read_csv(SEND_DIST)
    rec_dist   = pd.read_csv(REC_DIST)
    events_tt  = pd.read_csv(EVENTS)

    # ---------------------------------------------------------------
    # Überblick
    # ---------------------------------------------------------------
    print("=== Überblick ===")
    print("Events mit Topics:", len(events_tt))
    print("Kanten (Sender→Empfänger x Topic):", len(flow))
    print("Kanten (gesamt, aggregiert):", len(flow_total))
    print("Sender-Level x Topic:", len(send_dist))
    print("Recipient-Level x Topic:", len(rec_dist))

    # ---------------------------------------------------------------
    # Spalten robust erkennen
    # ---------------------------------------------------------------
    topic_cols = [c for c in ["thread_topic_id","topic_active","topic","Topic"] if c in flow.columns or c in events_tt.columns]
    topic_col = topic_cols[0] if topic_cols else None
    if not topic_col:
        print("\n[warn] Keine Topic-Spalte gefunden – überspringe Topic-Statistik.")
        return

    # ---------------------------------------------------------------
    # Topic-Statistik
    # ---------------------------------------------------------------
    print("\n=== Top 15 Topics nach Häufigkeit (Events) ===")
    col_ev = topic_col if topic_col in events_tt.columns else events_tt.columns[-1]
    print(events_tt[col_ev].value_counts().head(15).to_string())

    print("\n=== Top 10 Topics nach Gewicht (Flows) ===")
    col_fl = topic_col if topic_col in flow.columns else flow.columns[-1]
    print(flow.groupby(col_fl)["weight"].sum().sort_values(ascending=False).head(10).to_string())

    # ---------------------------------------------------------------
    # Beispielhafte Flüsse
    # ---------------------------------------------------------------
    print("\n=== Beispielhafte Flüsse (Top 10 by weight) ===")
    cols = [c for c in flow.columns if "sender_" in c or "recipient_" in c]
    if col_fl not in cols:
        cols += [col_fl]
    cols += ["weight"]
    print(flow.sort_values("weight", ascending=False).head(10)[cols].to_string(index=False))

    # ---------------------------------------------------------------
    # Dominante Sender-Level-Topics
    # ---------------------------------------------------------------
    print("\n=== Dominante Topics pro Sender-Level (Top 5 je Level) ===")
    lvl_col = send_dist.columns[0]
    for lvl, subdf in send_dist.groupby(lvl_col):
        print(f"\nSender-Level: {lvl}")
        col_sd = topic_col if topic_col in subdf.columns else subdf.columns[-3]
        print(subdf.sort_values("share", ascending=False).head(5)[[col_sd,"weight","share"]].to_string(index=False))

    # ---------------------------------------------------------------
    # Artefakt-Check
    # ---------------------------------------------------------------
    print("\n=== Artefakt-Check ===")
    label_cols = [c for c in ["Name","topic_name_active","topic_label"] if c in flow.columns]
    if label_cols:
        labels = " ".join(flow[label_cols[0]].dropna().astype(str).tolist())
        bad_tokens = [tok for tok in ["hou","ect","enronxgate","corp","gco","na","javaMail","thyme"]
                      if tok in labels.lower()]
        print("Gefundene verdächtige Tokens in Topic-Labels:", bad_tokens)
    else:
        print("Keine Label-Spalte gefunden (Labels evtl. nicht gemappt).")


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    diagnose_flows(env)