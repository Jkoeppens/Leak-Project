"""
pipe.inspect_metatopics
=======================
Überprüft die Ergebnisse der Meta-Topic-Reduktion:
- Verteilung der MetaTopics
- Zuordnung zu ursprünglichen Subtopics
- Beispieltexte pro MetaTopic

Beispiel:
---------
from pipe.setup_env import setup_environment
from pipe.inspect_metatopics import inspect_metatopics

env = setup_environment()
inspect_metatopics(env)
"""

from pathlib import Path
import pandas as pd


def inspect_metatopics(env: dict, examples_per_meta: int = 3) -> None:
    topics_dir = Path(env["outputs"]["topics_dir"])
    red_dir = topics_dir / "reduced"

    EV_META_CSV = red_dir / "events_with_metatopics.csv"
    MAP_CSV = red_dir / "topic_map_old_to_meta.csv"
    META_INFO = red_dir / "topic_info_meta.csv"
    TOPIC_INFO = topics_dir / "topic_info.csv"

    print("[paths check]")
    for p in [EV_META_CSV, MAP_CSV, META_INFO, TOPIC_INFO]:
        print(" -", p, "exists:", p.exists())

    evm = pd.read_csv(EV_META_CSV)
    map_tm = pd.read_csv(MAP_CSV)
    meta_info = pd.read_csv(META_INFO)
    ti = pd.read_csv(TOPIC_INFO)

    # --------------------------------------------------
    # 1) Verteilung der MetaTopics
    # --------------------------------------------------
    dist = evm["MetaTopic"].value_counts().sort_index()
    print("\n=== Verteilung MetaTopics (Events) ===")
    print(dist.to_string())

    # --------------------------------------------------
    # 2) MetaTopic -> Top-Subtopics
    # --------------------------------------------------
    ti2 = ti.rename(columns={"Topic": "topic_old", "Name": "name_old", "Count": "count_old"})
    mix = (
        map_tm.merge(ti2[["topic_old", "name_old", "count_old"]], on="topic_old", how="left")
        .groupby("topic_meta")
        .apply(
            lambda df: " | ".join(
                df.sort_values("count_old", ascending=False)
                .dropna(subset=["name_old"])
                .head(5)["name_old"]
                .astype(str)
            )
        )
        .rename("TopSubtopicNames")
        .reset_index()
    )

    meta_view = (
        meta_info.merge(mix, left_on="Topic", right_on="topic_meta", how="left")
        .drop(columns=["topic_meta"])
    )

    print("\n=== Meta-Topic Label + Top-Subtopic-Namen ===")
    print(meta_view[["Topic", "Name", "Count", "TopSubtopicNames"]].to_string(index=False))

    # --------------------------------------------------
    # 3) Beispiel-Events pro MetaTopic
    # --------------------------------------------------
    id_col = "event_id" if "event_id" in evm.columns else evm.columns[0]
    text_col = next(
        (c for c in ["text_full", "body_text", "text", "body"] if c in evm.columns), None
    )

    if text_col:
        print("\n=== Beispiele (pro MetaTopic) ===")
        for mt in sorted(evm["MetaTopic"].dropna().unique()):
            subs = evm[evm["MetaTopic"] == mt].head(examples_per_meta)
            print(f"\nMetaTopic {int(mt)} — {len(subs)} Beispiele:")
            for _, r in subs.iterrows():
                txt = str(r[text_col])[:300].replace("\n", " ")
                print(f" - {r[id_col]} | {txt}…")
    else:
        print(
            "\n[Hinweis] Keine Textspalte in evm gefunden "
            "(erwartet eine von: text_full/body_text/text/body)."
        )


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    inspect_metatopics(env)