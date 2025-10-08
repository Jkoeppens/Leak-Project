"""
pipe.topic_flows
================
Verknüpft Events, Topics/MetaTopics und Hierarchie-Levels zu Kommunikationsflüssen
zwischen Organisationseinheiten.  Erwartet die vorherigen Pipeline-Ergebnisse:

- events_clean.csv, events_roots_for_topics.csv
- events_topics_active.csv  (aus route_topics)
- infomap_levels.csv        (hierarchische Ebenen pro E-Mail)
- topic_info_* / labels_*   (Labels aus Topic-Modellen)

Erzeugt im reports/orgchart/:
  - topic_flow_edges_by_level.csv
  - topic_flow_edges_total.csv
  - topic_distribution_by_sender_level.csv
  - topic_distribution_by_recipient_level.csv
  - events_with_thread_topics.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re, json


def topic_flows(env: dict) -> None:
    # ---------------------------------------------------------------
    # Pfade aus config
    # ---------------------------------------------------------------
    root_dir = Path(env["paths"]["root"])
    clean_dir = Path(env["paths"]["clean_dir"])
    topics_dir = Path(env["outputs"]["topics_dir"])
    org_dir = Path(env["outputs"]["org_dir"])
    org_dir.mkdir(parents=True, exist_ok=True)

    levels_csv_path = root_dir / "data_derived" / "infomap_levels.csv"
    CLEAN_ALL = clean_dir / "events_clean.csv"
    ROOTS_FOR_TOPICS = clean_dir / "events_roots_for_topics.csv"
    TOPIC_EVENTS_ACT = topics_dir / "reduced" / "events_topics_active.csv"

    LABELS_META_PREF = topics_dir / "reduced" / "topic_info_meta_labeled.csv"
    LABELS_META_AUTO = topics_dir / "reduced" / "meta_labels_auto.csv"
    LABELS_ORIG = topics_dir / "topic_info_labels.csv"
    TOPIC_INFO_ORIG = topics_dir / "topic_info.csv"

    for p in [CLEAN_ALL, ROOTS_FOR_TOPICS, TOPIC_EVENTS_ACT, levels_csv_path]:
        assert p.exists(), f"Datei fehlt: {p}"

    # ---------------------------------------------------------------
    # Helper
    # ---------------------------------------------------------------
    EMAIL_RE = re.compile(r'<?([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})>?', re.I)

    def norm_email(x):
        if pd.isna(x):
            return None
        s = str(x).strip().lower()
        m = EMAIL_RE.search(s)
        return m.group(1) if m else None

    def parse_list_maybe_json_safe(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return []
        if isinstance(v, (list, tuple, set)):
            return list(v)
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    return json.loads(s)
                except Exception:
                    pass
            parts = re.split(r"[;, ]+", s)
            return [p for p in parts if "@" in p]
        return []

    def to_norm_sorted_unique_list(v):
        out = []
        for item in parse_list_maybe_json_safe(v):
            e = norm_email(item)
            if isinstance(e, str) and e:
                out.append(e)
        return sorted(set(out))

    # ---------------------------------------------------------------
    # Daten laden
    # ---------------------------------------------------------------
    events_all = pd.read_csv(CLEAN_ALL)
    roots_df = pd.read_csv(ROOTS_FOR_TOPICS)
    ev_topics = pd.read_csv(TOPIC_EVENTS_ACT)
    levels = pd.read_csv(levels_csv_path)

    # Vereinheitliche Schlüssel
    eid_col = next((c for c in events_all.columns if c.lower() in {"event_id","id","eid"}), None)
    events_all = events_all.rename(columns={eid_col: "event_id"})
    assert {"is_thread_root","thread_id"}.issubset(events_all.columns)

    if "event_id" not in ev_topics.columns:
        e_eid = next((c for c in ev_topics.columns if c.lower() in {"event_id","id","eid"}), None)
        ev_topics = ev_topics.rename(columns={e_eid:"event_id"})
    assert "topic_active" in ev_topics.columns

    # ---------------------------------------------------------------
    # Thread-Topics propagieren
    # ---------------------------------------------------------------
    root_topics = ev_topics[["event_id","topic_active"]].copy()
    thread_roots = events_all.loc[events_all["is_thread_root"], ["thread_id","event_id"]] \
                             .rename(columns={"event_id":"root_event_id"})
    thread_topic = (
        thread_roots.merge(root_topics, left_on="root_event_id", right_on="event_id", how="left")
                    .drop(columns=["event_id"])
                    .rename(columns={"topic_active":"thread_topic_active"})
    )
    events_all = events_all.merge(thread_topic, on="thread_id", how="left")

    # ---------------------------------------------------------------
    # Email-Felder normalisieren
    # ---------------------------------------------------------------
    events_all["from_email_norm"] = events_all.get("from_email", np.nan).map(norm_email)
    for col in ("to_emails","cc_emails","bcc_emails"):
        if col not in events_all.columns:
            events_all[col] = [[] for _ in range(len(events_all))]
        events_all[col + "_norm"] = events_all[col].apply(to_norm_sorted_unique_list)

    # ---------------------------------------------------------------
    # Level-Map vorbereiten
    # ---------------------------------------------------------------
    level_key = None
    for cand in ["email","mail","node","address","user","person"]:
        mask = levels.columns.str.lower() == cand
        if mask.any():
            level_key = levels.columns[mask][0]
            break
    if level_key is None:
        level_key = levels.columns[0]

    levels["_key_email"] = levels[level_key].map(norm_email)
    level_cols = [c for c in levels.columns if c not in (level_key, "_key_email")]
    level_map = levels.set_index("_key_email")[level_cols]

    def email_to_levels(email):
        if not isinstance(email, str) or not email:
            return {c: np.nan for c in level_cols}
        try:
            row = level_map.loc[email]
            if isinstance(row, pd.Series):
                return row.to_dict()
            else:
                return row.iloc[0].to_dict()
        except KeyError:
            return {c: np.nan for c in level_cols}

    sender_levels = events_all["from_email_norm"].apply(email_to_levels).apply(pd.Series)
    sender_levels.columns = [f"sender_{c}" for c in sender_levels.columns]
    events_all = pd.concat([events_all, sender_levels], axis=1)

    # ---------------------------------------------------------------
    # Empfänger-Kanten erzeugen
    # ---------------------------------------------------------------
    def recipients_records(row):
        recs = []
        for role, lst in [("TO", row["to_emails_norm"]),
                          ("CC", row["cc_emails_norm"]),
                          ("BCC", row["bcc_emails_norm"])]:
            for em in lst:
                lv = email_to_levels(em)
                rec = {
                    "event_id": row["event_id"],
                    "thread_id": row["thread_id"],
                    "role": role,
                    "recipient_email": em,
                    "topic_active": row["thread_topic_active"],
                    "is_newsletter": row.get("is_newsletter", False),
                }
                for c in sender_levels.columns:
                    rec[c] = row[c]
                for c in level_cols:
                    rec[f"recipient_{c}"] = lv.get(c, np.nan)
                recs.append(rec)
        return recs

    edge_rows = []
    for _, r in events_all.iterrows():
        edge_rows.extend(recipients_records(r))
    edges = pd.DataFrame(edge_rows)

    edges = edges[~edges["topic_active"].isna()]
    if "is_newsletter" in edges.columns:
        edges = edges[edges["is_newsletter"] == False]

    # ---------------------------------------------------------------
    # Aggregationen
    # ---------------------------------------------------------------
    LEVEL_DIM = level_cols[0]
    s_col, r_col = f"sender_{LEVEL_DIM}", f"recipient_{LEVEL_DIM}"
    edges["w"] = 1.0

    flow = (edges.groupby([s_col, r_col, "topic_active"], dropna=False)["w"]
                  .sum().reset_index().rename(columns={"w":"weight"}))
    flow_total = (edges.groupby([s_col, r_col], dropna=False)["w"]
                        .sum().reset_index().rename(columns={"w":"weight_total"}))

    sender_topic = (edges.groupby([s_col, "topic_active"], dropna=False)["w"]
                          .sum().reset_index(name="weight"))
    recipient_topic = (edges.groupby([r_col, "topic_active"], dropna=False)["w"]
                             .sum().reset_index(name="weight"))

    def add_share(df, levelcol):
        total = df.groupby(levelcol)["weight"].sum().rename("total")
        out = df.merge(total, on=levelcol, how="left")
        out["share"] = (out["weight"] / out["total"]).fillna(0.0)
        return out.drop(columns=["total"])

    sender_topic = add_share(sender_topic, s_col)
    recipient_topic = add_share(recipient_topic, r_col)

    # ---------------------------------------------------------------
    # Topic-Labels mergen
    # ---------------------------------------------------------------
    def load_label_table() -> pd.DataFrame:
        if LABELS_META_PREF.exists():
            ti = pd.read_csv(LABELS_META_PREF).rename(columns={"MetaTopic":"topic_active","label":"topic_name_active"})
            return ti[["topic_active","topic_name_active"]].drop_duplicates()
        if LABELS_META_AUTO.exists():
            ti = pd.read_csv(LABELS_META_AUTO)
            col_id = "MetaTopic" if "MetaTopic" in ti.columns else "topic_meta"
            col_lab = "label_1" if "label_1" in ti.columns else next(c for c in ti.columns if c.startswith("label"))
            ti = ti.rename(columns={col_id:"topic_active", col_lab:"topic_name_active"})
            return ti[["topic_active","topic_name_active"]].drop_duplicates()
        if LABELS_ORIG.exists():
            ti = pd.read_csv(LABELS_ORIG)
            if "topic_label" in ti.columns:
                ti = ti.rename(columns={"Topic":"topic_active","topic_label":"topic_name_active"})
            else:
                ti = ti.rename(columns={"Topic":"topic_active","Name":"topic_name_active"})
            return ti[["topic_active","topic_name_active"]].drop_duplicates()
        if TOPIC_INFO_ORIG.exists():
            ti = pd.read_csv(TOPIC_INFO_ORIG).rename(columns={"Topic":"topic_active","Name":"topic_name_active"})
            return ti[["topic_active","topic_name_active"]].drop_duplicates()
        return pd.DataFrame({"topic_active":[],"topic_name_active":[]})

    ti = load_label_table()

    def with_labels(df):
        if len(ti) == 0:
            return df
        return df.merge(ti, on="topic_active", how="left")

    flow_labeled = with_labels(flow)
    sender_topic_labeled = with_labels(sender_topic)
    recipient_topic_labeled = with_labels(recipient_topic)

    # ---------------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------------
    flow_labeled.to_csv(org_dir / "topic_flow_edges_by_level.csv", index=False)
    flow_total.to_csv(org_dir / "topic_flow_edges_total.csv", index=False)
    sender_topic_labeled.to_csv(org_dir / "topic_distribution_by_sender_level.csv", index=False)
    recipient_topic_labeled.to_csv(org_dir / "topic_distribution_by_recipient_level.csv", index=False)
    events_all[["event_id","thread_id","thread_topic_active"]].to_csv(
        org_dir / "events_with_thread_topics.csv", index=False
    )

    print("\n=== Flow-Beispiel (erste 8 Kanten) ===")
    print(flow_labeled.head(8).to_string(index=False))
    print("\n=== Top 10 Sender-Level x Topic ===")
    print(sender_topic_labeled.sort_values("weight", ascending=False).head(10).to_string(index=False))
    print("\n[write]", org_dir / "topic_flow_edges_by_level.csv")
    print("[write]", org_dir / "topic_flow_edges_total.csv")
    print("[write]", org_dir / "topic_distribution_by_sender_level.csv")
    print("[write]", org_dir / "topic_distribution_by_recipient_level.csv")
    print("[write]", org_dir / "events_with_thread_topics.csv")


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    topic_flows(env)