# === pipe/topics_flow/topic_flows.py ===
"""
Erzeugt gerichtete Topic-Flows zwischen organisationalen Clustern
basierend auf Ereignissen (z. B. E-Mail-Threads oder Meetings).

Input:
  - cluster_topic_pies_summary.csv
  - topic_distribution_by_level.csv
  - infomap_levels.csv
  - (optional) events_with_topics.csv mit Datumsspalte für Rolling Window

Output:
  - topic_flows_filtered.csv (bereit für Visualisierung)

Optional:
  - `period` Parameter (z. B. '2001-Q1', '2002', '2000-01-01_2000-03-31')
    um Rolling-Zeiträume zu erzeugen oder Teilmengen zu filtern.
"""

import pandas as pd
from pathlib import Path
import numpy as np

# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------

def load_label_table(env):
    """Lädt Topic-Labels aus topic_info_labels.csv im modernen oder alten Format."""
    topics_dir = Path(env["outputs"]["topics_dir"])
    ti_path = topics_dir / "topic_info_labels.csv"
    assert ti_path.exists(), f"❌ Datei fehlt: {ti_path}"

    ti = pd.read_csv(ti_path)
    cols = set(ti.columns)

    if {"Topic", "Name"}.issubset(cols):
        df = ti[["Topic", "Name"]].rename(
            columns={"Topic": "topic_active", "Name": "topic_name_active"}
        ).drop_duplicates()
    elif {"topic_active", "topic_name_active"}.issubset(cols):
        df = ti[["topic_active", "topic_name_active"]].drop_duplicates()
    else:
        raise ValueError(f"❌ Unbekannte Spalten in topic_info_labels.csv: {list(cols)}")

    print(f"[ok] Loaded topic label table with {len(df)} rows.")
    return df


def load_flows_input(env):
    """Lädt Pies und Hierarchieinformationen."""
    org_dir = Path(env["outputs"]["org_dir"])
    pies_summary = pd.read_csv(org_dir / "cluster_topic_pies_summary.csv")
    levels_csv = Path(env["paths"]["levels_csv"])
    levels = pd.read_csv(levels_csv)
    return pies_summary, levels


def ensure_datetime(df, col="date"):
    """Versucht, eine Datumsspalte sicher zu konvertieren."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def filter_by_period(df, period=None, date_col="date"):
    """Filtert ein DataFrame nach einem Zeitfenster oder Quartal."""
    if not period or date_col not in df.columns:
        return df

    if "_" in period:  # expliziter Bereich: "YYYY-MM-DD_YYYY-MM-DD"
        start, end = period.split("_")
        mask = (df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))
        return df.loc[mask]
    elif "-" in period and len(period) == 7:  # Jahr-Monat
        year, month = period.split("-")
        return df[df[date_col].dt.to_period("M") == f"{year}-{month}"]
    elif "-" not in period and len(period) == 4:  # Jahr
        return df[df[date_col].dt.year == int(period)]
    elif "Q" in period:  # Quartal
        return df[df[date_col].dt.to_period("Q").astype(str) == period]
    return df


# -----------------------------------------------------------------------------
# Hauptfunktion
# -----------------------------------------------------------------------------

def topic_flows(env: dict, period: str | None = None) -> None:
    print(f"\n=== [topic_flows] Start (period={period}) ===")

    org_dir = Path(env["outputs"]["org_dir"])
    topics_dir = Path(env["outputs"]["topics_dir"])
    levels_csv = Path(env["paths"]["levels_csv"])

    pies_summary_path = org_dir / "cluster_topic_pies_summary.csv"
    assert pies_summary_path.exists(), f"❌ Datei fehlt: {pies_summary_path}"

    flows_out = org_dir / "topic_flows_filtered.csv"

    # ---- Lade Daten ----
    pies_summary, levels = load_flows_input(env)
    label_table = load_label_table(env)

    # ---- Cluster-IDs & Topics ----
    flows = pies_summary[["cluster_id", "thread_topic_id", "weight", "n_events"]].copy()
    flows.rename(columns={"cluster_id": "sender_cluster"}, inplace=True)

    # Dummy-Empfänger auf gleicher Ebene simulieren (wenn keine echte Interaktion vorhanden)
    flows["recipient_cluster"] = flows["sender_cluster"]
    flows = flows.merge(label_table, left_on="thread_topic_id", right_on="topic_active", how="left")

    # ---- Falls Rolling Period aktiv ----
    ev_path = topics_dir / "events_with_topics.csv"
    if ev_path.exists():
        ev = pd.read_csv(ev_path)
        ev = ensure_datetime(ev, "date")
        ev = filter_by_period(ev, period, "date")
        print(f"[filter] Events gefiltert auf {len(ev)} Zeilen für period={period}")
    else:
        print("[warn] Keine events_with_topics.csv gefunden – period-Filter übersprungen.")

    # ---- Edge-Tabelle ----
    edges = flows.groupby(
        ["sender_cluster", "recipient_cluster", "thread_topic_id"], as_index=False
    ).agg({"weight": "sum", "n_events": "sum"})

    edges["color"] = "#1f77b4"
    edges["width"] = np.maximum(1.0, edges["weight"] / 10000.0)
    edges["edge_id"] = (
        edges["sender_cluster"].astype(str)
        + "||"
        + edges["recipient_cluster"].astype(str)
        + "||"
        + edges["thread_topic_id"].astype(str)
    )

    print(f"[ok] {len(edges)} edges erzeugt | Spalten: {list(edges.columns)}")

    # ---- Speichern ----
    flows_out.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(flows_out, index=False)
    print(f"[save] {flows_out}")

    print("✅ [topic_flows] abgeschlossen.")


# -----------------------------------------------------------------------------
# CLI Entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment

    env = setup_environment()
    topic_flows(env)