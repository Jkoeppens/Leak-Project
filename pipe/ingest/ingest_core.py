# ============================================================
# pipe/ingest/ingest_core.py
# Konsolidierte Ingest-Stufe für Leak-Project
# ============================================================

from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import hashlib, re, os

# --- Klassifikationslogik importieren (eigene Datei)
from pipe.ingest.flag_mail_types import flag_mail_types

# --- Schema: robuste Spaltenstruktur
COLUMNS = [
    "event_id", "thread_id", "timestamp", "date_local",
    "sender", "recipients_to", "recipients_cc", "n_recipients_total",
    "subject", "body_text", "text_length",
    "topics", "dominant_topic", "topic_labels",
    "cluster_id", "cluster_level", "leader",
    "in_degree", "out_degree", "centrality_zscore",
    "source_file", "parse_status", "ingest_timestamp",
    "content_type", "include_in_analysis"
]

# ------------------------------------------------------------
# Hauptfunktion
# ------------------------------------------------------------
def ingest_core(cfg, sample_limit=None):
    raw_dir = Path(cfg["paths"]["raw_dir"])
    clean_dir = Path(cfg["paths"]["clean_dir"])
    clean_dir.mkdir(parents=True, exist_ok=True)

    output_path = clean_dir / "events_master.csv"

    print(f"[Ingest] Using RAW dir: {raw_dir}")
    files = list(raw_dir.rglob("*"))
    if sample_limit:
        files = files[:sample_limit]
    print(f"[Ingest] {len(files)} Dateien gefunden")

    records = []
    for path in tqdm(files, desc="Parsing mails"):
        if not path.is_file():
            continue
        try:
            with open(path, "r", errors="ignore") as f:
                text = f.read()
            if not text.strip():
                continue

            msg_id = hashlib.md5(text.encode("utf-8")).hexdigest()
            sender_match = re.search(r"From:\s*(.*)", text)
            subj_match = re.search(r"Subject:\s*(.*)", text)
            date_match = re.search(r"Date:\s*(.*)", text)
            body = text.split("\n\n", 1)[-1]

            record = {
                "event_id": msg_id,
                "thread_id": hashlib.md5(
                    (sender_match.group(1) if sender_match else "").encode()
                ).hexdigest(),
                "timestamp": pd.to_datetime(date_match.group(1), errors="coerce", utc=True)
                if date_match else pd.NaT,
                "date_local": None,
                "sender": sender_match.group(1).strip() if sender_match else None,
                "recipients_to": None,
                "recipients_cc": None,
                "n_recipients_total": None,
                "subject": subj_match.group(1).strip() if subj_match else None,
                "body_text": body.strip(),
                "text_length": len(body),
                "topics": None,
                "dominant_topic": None,
                "topic_labels": None,
                "cluster_id": None,
                "cluster_level": None,
                "leader": None,
                "in_degree": None,
                "out_degree": None,
                "centrality_zscore": None,
                "source_file": str(path.relative_to(raw_dir)),
                "parse_status": "ok",
                "ingest_timestamp": datetime.now(datetime.UTC).isoformat(),
            }
            records.append(record)
        except Exception:
            records.append({
                "event_id": hashlib.md5(str(path).encode()).hexdigest(),
                "source_file": str(path.relative_to(raw_dir)),
                "parse_status": "failed"
            })

    df = pd.DataFrame(records, columns=COLUMNS).fillna({"parse_status": "ok"})

    # --- Typisierung integrieren
    df = flag_mail_types(df)
    df["include_in_analysis"] = ~df["content_type"].isin(
        ["newsletter", "empty_or_stub", "attachment_dump"]
    )

    # --- Schreiben
    df.to_csv(output_path, index=False)
    print(f"[Ingest] {len(df)} Datensätze → {output_path}")
    return output_path