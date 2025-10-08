"""
pipe.route_topics
=================
Verknüpft Events mit dem aktuell aktiven Topic-Level:
- Wenn USE_META=True: verwendet MetaTopics (+ Auto-Labels)
- Wenn USE_META=False: verwendet Original-Topics

Erzeugt:
  - events_topics_active_meta.csv
  - events_topics_active_orig.csv
  - events_topics_active.csv  (Alias für nachfolgende Pipeline)

Beispiel:
---------
from pipe.setup_env import setup_environment
from pipe.route_topics import route_topics

env = setup_environment()
route_topics(env, use_meta=True)
"""

from pathlib import Path
import pandas as pd


def route_topics(env: dict, use_meta: bool = True) -> None:
    """Führt Original-Topics oder MetaTopics in ein einheitliches Format."""
    topics_dir = Path(env["outputs"]["topics_dir"])
    reduced_dir = topics_dir / "reduced"
    reduced_dir.mkdir(parents=True, exist_ok=True)

    EVT_TOPICS = topics_dir / "events_with_topics.csv"
    TOPIC_INFO = topics_dir / "topic_info.csv"
    EV_META = reduced_dir / "events_with_metatopics.csv"
    META_MAP = reduced_dir / "meta_topic_label_map.csv"
    META_INFO_L = reduced_dir / "topic_info_meta_labeled.csv"
    META_INFO = reduced_dir / "topic_info_meta.csv"

    assert EVT_TOPICS.exists(), f"{EVT_TOPICS} fehlt."
    ev_orig = pd.read_csv(EVT_TOPICS)
    ev_meta = pd.read_csv(EV_META) if use_meta and EV_META.exists() else pd.DataFrame()

    # ---------------------------------------------------------------
    # Original Topic Column
    # ---------------------------------------------------------------
    orig_topic_candidates = ["thread_topic_id", "topic_id", "Topic", "topic"]
    ORIG_TOPIC_COL = next((c for c in orig_topic_candidates if c in ev_orig.columns), None)
    assert ORIG_TOPIC_COL, f"Keine Topic-Spalte in {EVT_TOPICS} gefunden."

    # ---------------------------------------------------------------
    # Meta Topic Column
    # ---------------------------------------------------------------
    meta_col_cands = ["MetaTopic", "topic_meta", "meta_id"]
    META_COL = next((c for c in meta_col_cands if c in ev_meta.columns), None) if use_meta else None
    if use_meta:
        assert not ev_meta.empty, f"{EV_META} fehlt – bitte zuerst den Reduktions-Block laufen lassen."
        assert META_COL, f"In {EV_META} keine Meta-Topic-Spalte gefunden."

    # ---------------------------------------------------------------
    # Label Map laden
    # ---------------------------------------------------------------
    if use_meta:
        if META_MAP.exists():
            meta_label_map = pd.read_csv(META_MAP)
            id_col = (
                "MetaTopic"
                if "MetaTopic" in meta_label_map.columns
                else ("topic_meta" if "topic_meta" in meta_label_map.columns else None)
            )
            assert id_col and "primary_label" in meta_label_map.columns
            meta_label_map = meta_label_map[[id_col, "primary_label"]].rename(columns={id_col: "meta_id"})
            label_col = "primary_label"
        elif META_INFO_L.exists():
            tmp = pd.read_csv(META_INFO_L)
            meta_label_map = tmp[["Topic", "label"]].rename(columns={"Topic": "meta_id", "label": "primary_label"})
            label_col = "primary_label"
        else:
            tmp = pd.read_csv(META_INFO)
            meta_label_map = tmp[["Topic", "Name"]].rename(columns={"Topic": "meta_id", "Name": "primary_label"})
            label_col = "primary_label"

    # ---------------------------------------------------------------
    # Routing
    # ---------------------------------------------------------------
    if use_meta:
        evj = ev_orig.merge(ev_meta[["event_id", META_COL]], on="event_id", how="left")
        evj["topic_active"] = evj[META_COL]
        evj = evj.merge(meta_label_map.rename(columns={"meta_id": "topic_active"}), on="topic_active", how="left")
        evj = evj.rename(columns={label_col: "topic_name_active"})
        topic_source = "meta"
    else:
        evj = ev_orig.copy()
        evj["topic_active"] = evj[ORIG_TOPIC_COL]
        tinfo = (
            pd.read_csv(TOPIC_INFO)[["Topic", "Name"]]
            .rename(columns={"Topic": "topic_active", "Name": "topic_name_active"})
        )
        evj = evj.merge(tinfo, on="topic_active", how="left")
        topic_source = "orig"

    # ---------------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------------
    OUT_MAIN = reduced_dir / f"events_topics_active_{topic_source}.csv"
    OUT_ALIAS = reduced_dir / "events_topics_active.csv"

    evj.to_csv(OUT_MAIN, index=False)
    evj.to_csv(OUT_ALIAS, index=False)

    print(f"[write] {OUT_MAIN}  rows={len(evj)} | USE_META={use_meta}")
    print(f"[alias] {OUT_ALIAS} (für Pipeline)")
    print(evj[["event_id", "topic_active", "topic_name_active"]].head(3))


if __name__ == "__main__":
    from pipe.setup_env import setup_environment
    env = setup_environment()
    route_topics(env, use_meta=True)