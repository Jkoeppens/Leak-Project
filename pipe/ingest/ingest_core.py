# ============================================================
# pipe/ingest/ingest_core.py – vereinfachte Version
# ============================================================

from pathlib import Path
from datetime import datetime, UTC
import pandas as pd
import hashlib, re
from tqdm import tqdm

# ------------------------------------------------------------
# 1️⃣ Helper: Mail-Datei-Filter
# ------------------------------------------------------------
def _is_mail_file(p: Path) -> bool:
    """Akzeptiert nur Text-Mails ohne Endung, ignoriert Systemdateien."""
    return p.is_file() and not p.name.startswith(".") and p.suffix == ""

# ------------------------------------------------------------
# 2️⃣ Hauptfunktion
# ------------------------------------------------------------
def ingest_core(cfg, sample_limit=None):
    cfg_paths = cfg["paths"]
    raw_dir = Path(cfg_paths["raw_dir"])
    clean_dir = Path(cfg_paths["clean_dir"])
    clean_dir.mkdir(parents=True, exist_ok=True)
    out_path = clean_dir / "events_master.csv"

    print(f"[Ingest] RAW dir : {raw_dir}")
    print(f"[Ingest] CLEAN dir: {clean_dir}")

    files = [p for p in raw_dir.rglob("*") if _is_mail_file(p)]
    if sample_limit:
        files = files[:sample_limit]
    print(f"[Ingest] {len(files)} Dateien gefunden\n")

    records = []
    for path in tqdm(files, desc="Parsing mails"):
        try:
            text = path.read_text(errors="ignore")
            if not text.strip():
                continue

            sender = re.search(r"^From:\s*(.*)", text, re.MULTILINE)
            subject = re.search(r"^Subject:\s*(.*)", text, re.MULTILINE)
            date = re.search(r"^Date:\s*(.*)", text, re.MULTILINE)

            body = text.split("\n\n", 1)[-1].strip()
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
                "source_file": str(path.relative_to(raw_dir)),
                "parse_status": "ok",
                "ingest_timestamp": datetime.now(UTC).isoformat(),
            }
            records.append(record)

        except Exception as e:
            records.append({
                "event_id": hashlib.md5(str(path).encode()).hexdigest(),
                "source_file": str(path.relative_to(raw_dir)),
                "parse_status": f"failed: {e}"
            })

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"\n[Ingest] {len(df)} Datensätze → {out_path}")
    return out_path