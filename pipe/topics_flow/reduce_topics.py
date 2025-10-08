"""
pipe.reduce_topics
==================
Reduziert vorhandene BERTopic-Ergebnisse auf ~N Meta-Themen
mittels AgglomerativeClustering auf Wortvektor-Ebene.
Erzeugt:
  - reduced/topic_map_old_to_meta.csv
  - reduced/events_with_metatopics.csv
  - reduced/topic_info_meta.csv

Beispiel:
---------
from pipe.setup_env import setup_environment
from pipe.reduce_topics import reduce_topics_to_meta

env = setup_environment()
reduce_topics_to_meta(env, n_meta=10)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from typing import Dict, Any


# ---------------------------------------------------------------------
def reduce_topics_to_meta(env: Dict[str, Any], n_meta: int = 10) -> None:
    """Reduziert BERTopic-Topics zu n_meta Meta-Themen."""
    topics_dir = Path(env["outputs"]["topics_dir"])
    out_dir = topics_dir / "reduced"
    out_dir.mkdir(parents=True, exist_ok=True)

    EVT_TOPICS = topics_dir / "events_with_topics.csv"
    TOPIC_INFO = topics_dir / "topic_info.csv"
    TOPIC_WORDS = topics_dir / "topic_words.csv"

    print("[paths check]")
    for p in [EVT_TOPICS, TOPIC_INFO, TOPIC_WORDS]:
        print(" -", p, "exists:", p.exists())

    tw = pd.read_csv(TOPIC_WORDS)
    ti = pd.read_csv(TOPIC_INFO)
    ev = pd.read_csv(EVT_TOPICS)

    # ---------------- Spalten prüfen ----------------
    topic_cols_candidates = ["thread_topic_id", "topic_id", "Topic", "topic"]
    TOPIC_COL = next((c for c in topic_cols_candidates if c in ev.columns), None)
    assert TOPIC_COL, f"Keine Topic-Spalte gefunden. Kandidaten: {topic_cols_candidates}"

    id_cols_candidates = ["event_id", "doc_id", "row_id"]
    ID_COL = next((c for c in id_cols_candidates if c in ev.columns), None)
    assert ID_COL, f"Keine ID-Spalte gefunden. Kandidaten: {id_cols_candidates}"

    tw_cols = {c.lower(): c for c in tw.columns}
    COL_T = tw_cols.get("topic", "Topic")
    COL_W = tw_cols.get("word", "Word")
    COL_S = tw_cols.get("weight", None)
    if COL_S is None or COL_S not in tw.columns:
        tw["_w"] = 1.0
        COL_S = "_w"

    # ---------------- Matrix bauen ----------------
    M = tw.pivot_table(index=COL_T, columns=COL_W, values=COL_S, aggfunc="sum", fill_value=0.0)
    topics_sorted = M.index.tolist()
    M_norm = normalize(M.values, norm="l2", axis=1)

    # ---------------- Clustering ----------------
    dist = cosine_distances(M_norm)
    clu = AgglomerativeClustering(
        n_clusters=n_meta,
        metric="precomputed",
        linkage="average"
    )
    labels = clu.fit_predict(dist)

    map_old_to_meta = pd.DataFrame({
        "topic_old": topics_sorted,
        "topic_meta": labels
    })

    sizes = map_old_to_meta["topic_meta"].value_counts().sort_index()
    print("[cluster sizes] meta_id -> #subtopics")
    print(sizes.to_string())

    # ---------------- Meta-Namen ----------------
    ti2 = ti.rename(columns={"Topic": "topic_old", "Name": "name_old", "Count": "count_old"})
    meta_names = (
        map_old_to_meta.merge(ti2[["topic_old", "name_old", "count_old"]], on="topic_old", how="left")
        .sort_values(["topic_meta", "count_old"], ascending=[True, False])
        .groupby("topic_meta")
        .agg(
            Name=("name_old", lambda s: " + ".join(pd.Series(s).dropna().astype(str).head(3))),
            Count=("count_old", "sum"),
        )
        .reset_index()
    )

    def centroid_top_words(meta_id, k=5):
        idx = [i for i, t in enumerate(topics_sorted)
               if map_old_to_meta.iloc[i]["topic_meta"] == meta_id]
        if not idx:
            return ""
        centroid = M_norm[idx].mean(axis=0)
        topj = np.argsort(centroid)[::-1][:k]
        words = M.columns[topj].tolist()
        return " ".join(words)

    mask_empty = meta_names["Name"].isna() | (meta_names["Name"].astype(str).str.strip() == "")
    meta_names.loc[mask_empty, "Name"] = meta_names.loc[mask_empty, "topic_meta"].apply(
        lambda mid: centroid_top_words(mid, k=5)
    )

    # ---------------- Events mappen ----------------
    ev["_topic_old"] = ev[TOPIC_COL].astype(float)
    mp = dict(zip(map_old_to_meta["topic_old"].astype(float),
                  map_old_to_meta["topic_meta"].astype(int)))
    ev["MetaTopic"] = ev["_topic_old"].map(mp)

    # ---------------- Speichern ----------------
    map_old_to_meta.sort_values(["topic_meta", "topic_old"]).to_csv(
        out_dir / "topic_map_old_to_meta.csv", index=False)
    ev_out = ev.drop(columns=["_topic_old"])
    ev_out.to_csv(out_dir / "events_with_metatopics.csv", index=False)

    meta_info = meta_names.rename(columns={"topic_meta": "Topic"})
    meta_info = meta_info.sort_values("Count", ascending=False)
    meta_info.to_csv(out_dir / "topic_info_meta.csv", index=False)

    print(f"[write] {out_dir/'topic_map_old_to_meta.csv'}")
    print(f"[write] {out_dir/'events_with_metatopics.csv'}  rows={len(ev_out)}")
    print(f"[write] {out_dir/'topic_info_meta.csv'}  meta_topics={meta_info['Topic'].nunique()}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    reduce_topics_to_meta(env, n_meta=10)