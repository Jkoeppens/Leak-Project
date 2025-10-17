# ============================================================
# pipe/ingest/ingest_core.py
# Robuste Ingest-Stufe mit Typisierung & Audit-Hooks
# ============================================================

from pathlib import Path
from datetime import datetime, UTC
import pandas as pd
import hashlib, re, os
from tqdm import tqdm

from pipe.ingest.flag_mail_types import flag_mail_types


def _is_mail_file(p: Path):
    """Erkennt Text-Mails anhand einfacher Header-Heuristik."""
    if p.name.startswith("."):
        return False
    if p.suffix.lower() in {".mbox", ".eml", ".txt"}:
        return True
    if p.suffix == "":
        try:
            with open(p, "r", errors="ignore") as f:
                head = f.read(300)
            return any(tag in head for tag in ("From:", "Subject:", "Message-ID:"))
        except Exception:
            return False
    return False


def ingest_core(cfg, per_owner_limit=500, max_owners=None):
    """Ingest pro Benutzer, limitiert, mit Typisierung und Audit-Ausgabe."""
    raw_dir = Path(cfg["paths"]["raw_dir"])
    clean_dir = Path(cfg["paths"]["clean_dir"])
    clean_dir.mkdir(parents=True, exist_ok=True)
    output_path = clean_dir / "events_master.csv"

    owners = [p for p in raw_dir.iterdir() if p.is_dir()]
    if max_owners:
        owners = owners[:max_owners]

    records = []

    print(f"[Ingest] RAW dir : {raw_dir}")
    print(f"[Ingest] CLEAN dir: {clean_dir}")
    print(f"[Ingest] Scanning {len(owners)} owners...\n")

    for owner in tqdm(owners, desc="Owners"):
        count = 0
        for file in owner.rglob("*"):
            if count >= per_owner_limit:
                break
            if not _is_mail_file(file):
                continue

            try:
                text = file.read_text(errors="ignore")
                header, _, body = text.partition("\n\n")

                sender = re.search(r"From:\s*(.*)", header)
                subject = re.search(r"Subject:\s*(.*)", header)
                date = re.search(r"Date:\s*(.*)", header)

                record = {
                    "event_id": hashlib.md5(text.encode()).hexdigest(),
                    "thread_id": hashlib.md5((sender.group(1) if sender else "").encode()).hexdigest(),
                    "timestamp": pd.to_datetime(date.group(1), errors="coerce", utc=True) if date else pd.NaT,
                    "date_local": None,
                    "sender": sender.group(1).strip() if sender else None,
                    "recipients_to": None,
                    "recipients_cc": None,
                    "n_recipients_total": None,
                    "subject": subject.group(1).strip() if subject else None,
                    "body_text": body,
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
                    "source_file": str(file.relative_to(raw_dir)),
                    "parse_status": "ok",
                    "ingest_timestamp": datetime.now(UTC).isoformat(),
                }
                records.append(record)
                count += 1
            except Exception as e:
                records.append({
                    "event_id": hashlib.md5(str(file).encode()).hexdigest(),
                    "source_file": str(file.relative_to(raw_dir)),
                    "parse_status": f"failed: {e}"
                })

    df = pd.DataFrame(records)
    if df.empty:
        print("[Ingest] Keine Datensätze erzeugt.")
        df.to_csv(output_path, index=False)
        return output_path

    df = flag_mail_types(df)
    df.to_csv(output_path, index=False)
    print(f"\n[Ingest] {len(df)} Datensätze → {output_path}")
    print(df["content_type"].value_counts())

    return output_path