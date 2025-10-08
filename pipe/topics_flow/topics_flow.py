"""
pipe.topics_flow
================
Führt BERTopic auf bereinigten Thread-Root-E-Mails aus.
Erzeugt:
  - events_with_topics.csv
  - topic_info.csv
  - topic_words.csv
  - topic_info_labels.csv
  - topic_probabilities.npy (optional)

Beispiel:
---------
from pipe.setup_env import setup_environment
from pipe.topics_flow import run_topic_model

env = setup_environment()
run_topic_model(env, use_subset=False)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import umap, hdbscan, time
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------
# Parameter-Defaults
# ---------------------------------------------------------------------
N_NEIGHBORS = 15
N_COMPONENTS = 5
MIN_CLUSTER_SIZE = 15
MIN_SAMPLES = 10
MIN_DF = 2
MAX_DF = 0.95
EMBED_MODEL = "all-MiniLM-L6-v2"
RANDOM_STATE = 42
MIN_TOK_LEN = 12


# ---------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------
def label_from_info(name: str, n: int = 4) -> str:
    if not isinstance(name, str):
        return ""
    parts = [p.strip() for p in name.split() if p.strip()]
    return " ".join(parts[:n])


# ---------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------
def run_topic_model(
    env: Dict[str, Any],
    use_subset: bool = False,
    subset_n: Optional[int] = None,
    subset_frac: Optional[float] = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Führt BERTopic auf events_roots_for_topics.csv aus und speichert die Ergebnisse.
    """

    clean_dir = Path(env["paths"]["clean_dir"])
    topics_dir = Path(env["outputs"]["topics_dir"])
    PATCHED = clean_dir / "events_roots_for_topics.csv"

    assert PATCHED.exists(), f"Input fehlt: {PATCHED}"
    topics_dir.mkdir(parents=True, exist_ok=True)

    DF = pd.read_csv(PATCHED)
    if "text_clean" not in DF.columns:
        raise ValueError("Spalte 'text_clean' fehlt. Bitte Block 2 ausführen.")

    if "is_thread_root" in DF.columns:
        DF = DF[DF["is_thread_root"] == True].copy()
    if "is_newsletter" in DF.columns:
        DF = DF[DF["is_newsletter"] == False].copy()

    DF["text_clean"] = DF["text_clean"].astype(str)
    DF["tok_len"] = DF["text_clean"].str.findall(r"\w+").str.len()
    DF_fit = DF[DF["tok_len"] >= MIN_TOK_LEN].copy()

    # Subset-Optionen
    if use_subset:
        if subset_n is not None:
            DF_fit = DF_fit.sample(n=min(subset_n, len(DF_fit)), random_state=random_state)
        elif subset_frac is not None:
            DF_fit = DF_fit.sample(frac=float(subset_frac), random_state=random_state)
        DF_fit = DF_fit.sort_index()

    docs = DF_fit["text_clean"].tolist()
    print(f"[info] Thread-Roots fürs Topic-Modell: {len(docs)} von {len(DF)} Root-Mails")

    # -----------------------------------------------------------------
    # Modelle
    # -----------------------------------------------------------------
    embedding_model = SentenceTransformer(EMBED_MODEL)

    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=MIN_DF,
        max_df=MAX_DF,
    )

    umap_model = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=N_COMPONENTS,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=True,
    )

    # -----------------------------------------------------------------
    # Fit + Ergebnisse
    # -----------------------------------------------------------------
    t0 = time.time()
    topics, probs = topic_model.fit_transform(docs)
    print(f"[timing] fit_transform: {time.time() - t0:,.1f}s")

    topic_info = topic_model.get_topic_info()

    # Topic-Wortlisten
    rows = []
    for topic_id in topic_info["Topic"].tolist():
        if topic_id == -1:
            continue
        for w, s in (topic_model.get_topic(topic_id) or []):
            rows.append({"Topic": topic_id, "Word": w, "Score": float(s)})
    topic_words = pd.DataFrame(rows)

    # Max-Probability je Dokument
    prob_max = probs.max(axis=1) if isinstance(probs, np.ndarray) else np.full(len(DF_fit), np.nan)
    DF_fit = DF_fit.reset_index(drop=True)
    DF_fit["topic_id"] = topics
    DF_fit["topic_prob_max"] = prob_max

    DF_out = DF.merge(
        DF_fit[["event_id", "topic_id", "topic_prob_max"]],
        on="event_id", how="left",
    )

    # -----------------------------------------------------------------
    # Speichern
    # -----------------------------------------------------------------
    DF_out.to_csv(topics_dir / "events_with_topics.csv", index=False)
    topic_info.to_csv(topics_dir / "topic_info.csv", index=False)
    topic_words.to_csv(topics_dir / "topic_words.csv", index=False)

    topic_info2 = topic_info.copy()
    topic_info2["topic_label"] = topic_info2["Name"].map(lambda s: label_from_info(s, n=4))
    topic_info2.to_csv(topics_dir / "topic_info_labels.csv", index=False)

    if isinstance(probs, np.ndarray):
        np.save(topics_dir / "topic_probabilities.npy", probs)
        print(f"[write] {topics_dir/'topic_probabilities.npy'} shape={probs.shape}")

    print("[write]", topics_dir / "events_with_topics.csv")
    print("[write]", topics_dir / "topic_info.csv")
    print("[write]", topics_dir / "topic_words.csv")
    print("[write]", topics_dir / "topic_info_labels.csv")

    noise_row = topic_info.loc[topic_info["Topic"] == -1, "Count"]
    noise = int(noise_row.iloc[0]) if len(noise_row) else 0
    print(f"[summary] Topics (ohne Noise): {len(topic_info[topic_info['Topic']!=-1])}, Noise: {noise}")
    return DF_out


# ---------------------------------------------------------------------
if __name__ == "__main__":
    from pipe.setup_env import setup_environment
    env = setup_environment()
    run_topic_model(env, use_subset=False)