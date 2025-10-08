"""
pipe.cluster_topic_pies
=======================
Erzeugt Topic-Verteilungen pro organisationalem Cluster (z. B. Abteilung/Team)
basierend auf den in levels.csv definierten Hierarchieebenen.

Input:
  - events_clean.csv
  - events_with_thread_topics.csv  (aus topic_flows)
  - topic_info_labels.csv          (aus topic_model)
  - infomap_levels.csv             (aus Block 1)

Output (reports/orgchart/):
  - cluster_topic_pies.csv              — Rohverteilung
  - cluster_topic_pies_summary.csv      — Top-Topics je Cluster (z. B. für Torten)

Beispiel:
---------
from pipe.setup_env import setup_environment
from pipe.cluster_topic_pies import cluster_topic_pies

env = setup_environment()
cluster_topic_pies(env)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re, json


def cluster_topic_pies(env: dict) -> None:
    # ---- Settings ----
    EXCLUDE_NOISE = True
    MIN_TOK_LEN   = 12
    WEIGHT_BY     = "char"   # 'char' | 'tok' | 'count'
    ROOTS_ONLY    = False
    AUTO_PICK_LEVEL = True
    CLUSTER_LEVEL_COL = None
    TOP_N_PER_CLUSTER = 8

    # ---- Paths ----
    clean_dir  = Path(env["paths"]["clean_dir"])
    topics_dir = Path(env["outputs"]["topics_dir"])
    org_dir    = Path(env["outputs"]["org_dir"])
    org_dir.mkdir(parents=True, exist_ok=True)

    CLEAN_ALL          = clean_dir / "events_clean.csv"
    EVENTS_THREAD_TOPS = org_dir   / "events_with_thread_topics.csv"
    TOPIC_INFO_LABELS  = topics_dir / "topic_info_labels.csv"
    LEVELS_CSV_PATH    = Path(env["paths"]["levels_csv"])

    for p in [CLEAN_ALL, EVENTS_THREAD_TOPS, TOPIC_INFO_LABELS, LEVELS_CSV_PATH]:
        assert p.exists(), f"Datei fehlt: {p}"

    # ---- Helpers ----
    EMAIL_RE = re.compile(r'<?([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})>?', re.I)
    def norm_email(x):
        if pd.isna(x): return None
        s = str(x).strip().lower()
        m = EMAIL_RE.search(s)
        return m.group(1) if m else None

    # ---- Load ----
    ev = pd.read_csv(CLEAN_ALL)
    et = pd.read_csv(EVENTS_THREAD_TOPS)
    ti = pd.read_csv(TOPIC_INFO_LABELS)
    levels = pd.read_csv(LEVELS_CSV_PATH)

    eid_ev = next((c for c in ev.columns if c.lower() in {"event_id","id","eid"}), None)
    ev = ev.rename(columns={eid_ev:"event_id"})
    eid_et = next((c for c in et.columns if c.lower() in {"event_id","id","eid"}), None)
    et = et.rename(columns={eid_et:"event_id"})

    assert "thread_topic_id" in et.columns, "events_with_thread_topics.csv braucht 'thread_topic_id'."
    if "thread_topic_prob" not in et.columns: et["thread_topic_prob"] = np.nan
    ev = ev.merge(et[["event_id","thread_topic_id","thread_topic_prob"]], on="event_id", how="left")

    # ---- Filter ----
    if EXCLUDE_NOISE:
        ev = ev[ev["thread_topic_id"].notna() & (ev["thread_topic_id"] != -1)]
    else:
        ev = ev[ev["thread_topic_id"].notna()]
    if ROOTS_ONLY and "is_thread_root" in ev.columns:
        ev = ev[ev["is_thread_root"] == True]

    # ---- Lengths ----
    ev["char_len"] = ev.get("text_clean", "").astype(str).str.len()
    ev["tok_len"]  = ev.get("text_clean", "").astype(str).str.findall(r"\w+").str.len()
    ev = ev[ev["tok_len"] >= MIN_TOK_LEN].copy()

    # ---- Map Sender → Cluster ----
    level_key = None
    for cand in ["email","mail","node","address","user","person"]:
        mask = levels.columns.str.lower() == cand
        if mask.any():
            level_key = levels.columns[mask][0]; break
    if level_key is None: level_key = levels.columns[0]

    levels["_key_email"] = levels[level_key].map(norm_email)
    level_cols = [c for c in levels.columns if c not in (level_key, "_key_email")]
    assert len(level_cols)>=1, f"Keine Levelspalten in {LEVELS_CSV_PATH}. Gefunden: {list(levels.columns)}"

    if AUTO_PICK_LEVEL or not CLUSTER_LEVEL_COL:
        CLUSTER_LEVEL_COL = "module_path" if "module_path" in level_cols else level_cols[0]

    lvl = levels.dropna(subset=["_key_email"]).copy()
    lvl[CLUSTER_LEVEL_COL] = lvl[CLUSTER_LEVEL_COL].astype(str).fillna("")
    agg = lvl.groupby("_key_email", as_index=False).agg({CLUSTER_LEVEL_COL: lambda s: max(s, key=len)})
    level_map = agg.set_index("_key_email")[CLUSTER_LEVEL_COL]

    ev["from_email_norm"] = ev.get("from_email", np.nan).map(norm_email)
    ev["cluster_id"] = ev["from_email_norm"].map(level_map)
    coverage = ev["cluster_id"].notna().mean()
    print(f"[info] Cluster-Mapping Coverage: {coverage:.1%}  | Cluster-Spalte: {CLUSTER_LEVEL_COL}")
    ev = ev[ev["cluster_id"].notna()].copy()

    # ---- Gewicht ----
    if WEIGHT_BY == "char":
        ev["w"] = ev["char_len"].astype(float)
    elif WEIGHT_BY == "tok":
        ev["w"] = ev["tok_len"].astype(float)
    elif WEIGHT_BY == "count":
        ev["w"] = 1.0
    else:
        raise ValueError("WEIGHT_BY muss 'char', 'tok' oder 'count' sein.")

    # ---- Aggregation ----
    grp = ev.groupby(["cluster_id","thread_topic_id"], dropna=False).agg(
        weight=("w","sum"),
        n_events=("event_id","nunique"),
        avg_prob=("thread_topic_prob","mean"),
    ).reset_index()

    tot = grp.groupby("cluster_id")["weight"].sum().rename("weight_total")
    grp = grp.merge(tot, on="cluster_id", how="left")
    grp["share"] = np.where(grp["weight_total"]>0, grp["weight"]/grp["weight_total"], 0.0)

    # ---- Labels ----
    ti2 = ti.copy()
    if "Topic" in ti2.columns: ti2 = ti2.rename(columns={"Topic":"thread_topic_id"})
    elif "topic_id" in ti2.columns: ti2 = ti2.rename(columns={"topic_id":"thread_topic_id"})
    grp = grp.merge(ti2, on="thread_topic_id", how="left")

    # ---- Outputs ----
    OUT_PIES         = org_dir / "cluster_topic_pies.csv"
    OUT_PIES_SUMMARY = org_dir / "cluster_topic_pies_summary.csv"

    grp.to_csv(OUT_PIES, index=False)
    summary = (grp.sort_values(["cluster_id","weight"], ascending=[True, False])
                  .groupby("cluster_id").head(TOP_N_PER_CLUSTER).reset_index(drop=True))
    summary.to_csv(OUT_PIES_SUMMARY, index=False)

    print(f"[write] {OUT_PIES}  rows={len(grp)}")
    print(f"[write] {OUT_PIES_SUMMARY}  rows={len(summary)}")
    print(f"Gewicht: {WEIGHT_BY} | Noise ausgeschlossen: {EXCLUDE_NOISE} | MIN_TOK_LEN={MIN_TOK_LEN}")
    print("\nBeispiel:")
    print(grp.head(10)[['cluster_id','thread_topic_id','weight','share','n_events']].to_string(index=False))


if __name__ == "__main__":
    from pipe.setup_env import setup_environment
    env = setup_environment()
    cluster_topic_pies(env)