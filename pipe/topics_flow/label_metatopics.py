"""
pipe.label_metatopics
=====================
Erzeugt automatische Labels für MetaTopics via c-TF-IDF
und speichert mehrere Artefakte:
  - meta_labels_auto.csv          (alle Kandidaten)
  - meta_topic_label_map.csv      (primäre Labels)
  - topic_info_meta_labeled.csv   (MetaTopic-Infos mit Label)

Beispiel:
---------
from pipe.setup_env import setup_environment
from pipe.label_metatopics import label_metatopics

env = setup_environment()
label_metatopics(env)
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# ---------------------------------------------------------------------
def label_metatopics(env: dict, top_k: int = 8, top_show: int = 3) -> None:
    """Erzeugt automatische MetaTopic-Labels via c-TF-IDF."""
    topics_dir = Path(env["outputs"]["topics_dir"])
    reduced_dir = topics_dir / "reduced"

    EV_META_CSV = reduced_dir / "events_with_metatopics.csv"
    TOPIC_INFO_CS = reduced_dir / "topic_info_meta.csv"
    OUT_LABELS = reduced_dir / "meta_labels_auto.csv"
    OUT_MAP = reduced_dir / "meta_topic_label_map.csv"
    OUT_INFO_LBL = reduced_dir / "topic_info_meta_labeled.csv"

    ev = pd.read_csv(EV_META_CSV)
    ti = pd.read_csv(TOPIC_INFO_CS)

    # -----------------------------------------------------------------
    # Spalten finden
    # -----------------------------------------------------------------
    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(f"Keine der Spalten gefunden: {candidates}\nVorhanden: {list(df.columns)}")

    meta_col = find_col(ev, ["MetaTopic", "topic_meta", "meta_topic", "metatopic"])
    text_cols_candidates = ["text_clean", "body_text", "text_full", "subject"]
    text_cols = [c for c in text_cols_candidates if c in ev.columns]
    if not text_cols:
        raise KeyError(f"Keine Textspalte gefunden. Erwartet eine von: {text_cols_candidates}")

    print(f"[info] Meta-Spalte: {meta_col} | Textspalten genutzt: {text_cols}")

    # -----------------------------------------------------------------
    # Text vorbereiten
    # -----------------------------------------------------------------
    def clean_text(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"http[s]?://\\S+|www\\.\\S+", " ", s)
        s = re.sub(r"\\b[\\w\\.-]+@[\\w\\.-]+\\.\\w+\\b", " ", s)
        s = re.sub(r"\\b\\d{1,2}[:h]\\d{2}\\b", " ", s)
        s = re.sub(r"[_\\-=*]{2,}", " ", s)
        s = re.sub(r"\\s+", " ", s)
        return s

    ev["_text_for_label"] = ev[text_cols].astype(str).agg(" ".join, axis=1).map(clean_text)

    artifact_tokens = {
        "image", "images", "img", "gif", "click", "unsubscribe", "http", "https",
        "www", "hou", "ect", "enron", "corp", "na", "re", "fw", "fwd",
        "thyme", "evans", "com", "net", "org", "mailto", "cc", "bcc", "pm", "am"
    }

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
    )

    # -----------------------------------------------------------------
    # c-TF-IDF pro MetaTopic
    # -----------------------------------------------------------------
    grouped = ev.groupby(meta_col, as_index=True)["_text_for_label"].apply(lambda x: " ".join(x.tolist()))
    metas = grouped.index.tolist()
    docs = grouped.tolist()

    X = vectorizer.fit_transform(docs)
    tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
    X_tfidf = tfidf.fit_transform(X)
    vocab = np.array(vectorizer.get_feature_names_out())

    def is_artifact(phrase: str) -> bool:
        toks = [t for t in re.split(r"[^\w]+", phrase.lower()) if t]
        if not toks:
            return True
        if all(t in artifact_tokens for t in toks):
            return True
        if all(t.isdigit() for t in toks):
            return True
        return False

    # -----------------------------------------------------------------
    # Top-Phrasen je MetaTopic
    # -----------------------------------------------------------------
    rows = []
    for i, meta_id in enumerate(metas):
        row = X_tfidf[i].toarray().ravel()
        order = row.argsort()[::-1]
        cand = []
        for idx in order:
            phrase = vocab[idx]
            if is_artifact(phrase):
                continue
            phrase = re.sub(r"\\b\\d{2,}\\b", "", phrase).strip()
            phrase = re.sub(r"\\s+", " ", phrase)
            if len(phrase) < 3:
                continue
            cand.append(phrase)
            if len(cand) >= top_k:
                break
        top = cand[:top_show] if cand else []
        rows.append({
            "MetaTopic": int(meta_id) if pd.notna(meta_id) else None,
            "label_1": top[0] if len(top) > 0 else "",
            "label_2": top[1] if len(top) > 1 else "",
            "label_3": top[2] if len(top) > 2 else "",
            "candidates": " | ".join(cand),
        })

    labels_df = pd.DataFrame(rows).sort_values("MetaTopic").reset_index(drop=True)
    labels_df.to_csv(OUT_LABELS, index=False)
    print(f"[write] {OUT_LABELS}  rows={len(labels_df)}")

    # -----------------------------------------------------------------
    # Primary Label auswählen
    # -----------------------------------------------------------------
    def pick_primary(row):
        for c in ["label_1", "label_2", "label_3"]:
            ph = str(row.get(c, "")).strip()
            if not ph:
                continue
            wc = len(ph.split())
            if 2 <= wc <= 3:
                return ph
        for c in ["label_1", "label_2", "label_3"]:
            ph = str(row.get(c, "")).strip()
            if ph:
                return ph
        return f"MetaTopic {row['MetaTopic']}"

    label_map = labels_df[["MetaTopic", "label_1", "label_2", "label_3"]].copy()
    label_map["primary_label"] = label_map.apply(pick_primary, axis=1)
    label_map = label_map[["MetaTopic", "primary_label"]]
    label_map.to_csv(OUT_MAP, index=False)
    print(f"[write] {OUT_MAP}")

    # -----------------------------------------------------------------
    # topic_info_meta mergen
    # -----------------------------------------------------------------
    ti_col = "Topic" if "Topic" in ti.columns else find_col(ti, ["MetaTopic", "topic", "meta_topic"])
    ti_labeled = ti.merge(label_map, left_on=ti_col, right_on="MetaTopic", how="left")
    ti_labeled = ti_labeled.drop(columns=["MetaTopic"])
    ti_labeled.to_csv(OUT_INFO_LBL, index=False)
    print(f"[write] {OUT_INFO_LBL}")

    # -----------------------------------------------------------------
    # Vorschau
    # -----------------------------------------------------------------
    print("\n=== Auto-Labels (Preview) ===")
    print(labels_df.head(10).to_string(index=False))


# ---------------------------------------------------------------------
if __name__ == "__main__":
    from pipe.setup_env import setup_environment
    env = setup_environment()
    label_metatopics(env)