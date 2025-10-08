"""
pipe.analyze_topics
===================
Lädt die Ausgabedateien aus pipe.topics_flow und erzeugt eine
textuelle Übersicht über Topic-Statistiken, häufige Wörter
und Beispieldokumente.

Beispiel:
---------
from pipe.setup_env import setup_environment
from pipe.analyze_topics import analyze_topics

env = setup_environment()
analyze_topics(env)
"""

from pathlib import Path
import pandas as pd


def analyze_topics(env: dict, top_n: int = 10, examples_per_topic: int = 3) -> None:
    """Analysiert die BERTopic-Ergebnisse im topics_dir."""
    topics_dir = Path(env["outputs"]["topics_dir"])
    topic_info = pd.read_csv(topics_dir / "topic_info.csv")
    words = pd.read_csv(topics_dir / "topic_words.csv")
    events = pd.read_csv(topics_dir / "events_with_topics.csv")

    # --- Spalten robust bestimmen ---
    topic_cols_candidates = [
        "Topic", "topic", "topic_id", "thread_topic_id",
        "topic_id_root", "topic_root"
    ]
    topic_col = next((c for c in topic_cols_candidates if c in events.columns), None)
    if topic_col is None:
        raise ValueError(
            f"Keine Topic-Spalte gefunden. Kandidaten: {topic_cols_candidates}\n"
            f"Vorhanden: {list(events.columns)}"
        )

    text_cols_candidates = ["text", "text_full", "text_clean", "body", "content"]
    text_col = next((c for c in text_cols_candidates if c in events.columns), None)
    if text_col is None:
        raise ValueError(
            f"Keine Text-Spalte gefunden. Kandidaten: {text_cols_candidates}\n"
            f"Vorhanden: {list(events.columns)}"
        )

    # --- Übersicht/Statistik ---
    print("=== Topics Übersicht ===")
    n_topics = (topic_info["Topic"] != -1).sum()
    n_noise_rows = (topic_info["Topic"] == -1).sum()
    print(f"Topics ohne Noise: {n_topics}")
    print(f"Noise-Zeilen in topic_info (-1): {n_noise_rows}")
    print(f"Gesamt-Zeilen in topic_info: {len(topic_info)}")

    print("\n=== Top Topics (nach Dokumentanzahl) ===")
    top_topics = topic_info[topic_info["Topic"] != -1].nlargest(top_n, "Count")
    print(top_topics[["Topic", "Count", "Name"]])

    topic_counts = topic_info[topic_info["Topic"] != -1]["Count"]
    print("\n=== Verteilungs-Statistik (topic_info) ===")
    print(f"Ø Dokumente/Topic: {topic_counts.mean():.1f}")
    print(f"Median: {topic_counts.median()} | Max: {topic_counts.max()} | Min: {topic_counts.min()}")

    # --- Beispiel-Wörter pro Topic ---
    print("\n=== Beispiel-Wörter pro Topic ===")
    for tid in top_topics["Topic"]:
        w = (
            words[words["Topic"] == tid]["Word"]
            .dropna().astype(str).head(10).tolist()
        )
        print(f"Topic {tid}: {', '.join(w)}")

    # --- Beispiel-Dokumente pro Topic ---
    print("\n=== Beispiel-Dokumente pro Topic ===")
    for tid in top_topics["Topic"].head(3):
        sub = (
            events[events[topic_col] == tid][text_col]
            .dropna().astype(str).head(examples_per_topic).tolist()
        )
        print(f"\nTopic {tid} – {len(sub)} Beispiele:")
        for d in sub:
            print(" -", (d[:240] + "…") if len(d) > 240 else d)

    # --- Cross-Check: Topic Counts ---
    print("\n=== Cross-Check: Counts aus events_with_topics ===")
    ev_counts = (
        events[topic_col].value_counts()
        .rename_axis("Topic")
        .reset_index(name="Count")
    )
    print(ev_counts.head(10))


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    analyze_topics(env)