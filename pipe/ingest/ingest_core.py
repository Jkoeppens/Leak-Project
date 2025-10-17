# ============================================================
# pipe/ingest/ingest_core.py
# Konsolidierte, audit-kompatible Ingest-Stufe für Leak-Project
# Branch: consolidate/ingest-core
# ============================================================

from pathlib import Path
from datetime import datetime, UTC
import pandas as pd
import hashlib, re, os
from tqdm import tqdm

# --- Klassifikationslogik importieren
try:
    from pipe.ingest.flag_mail_types import flag_mail_types
except ImportError:
    flag_mail_types = None  # Fallback für isolierte Tests

# ============================================================
# 1️⃣ Helper: {root}-Placeholder-Absicherung
# ============================================================
def _resolve_paths_inplace(cfg: dict):
    """Ersetzt {root} in allen bekannten Sektionen sicher."""
    if "paths" not in cfg or "root" not in cfg["paths"]:
        return cfg
    root = cfg["paths"]["root"]
    for sec in ("paths", "outputs", "reports", "ingest"):
        if sec in cfg:
            for k, v in list(cfg[sec].items()):
                if isinstance(v, str) and "{root}" in v:
                    cfg[sec][k] = v.replace("{root}", root)
    return cfg


# ============================================================
# 2️⃣ Hauptfunktion: ingest_core
# ============================================================
def ingest_core(cfg, sample_limit=None):
    """
    Liest Mails aus data_raw, extrahiert Header + Body,
    erzeugt events_master.csv inkl. Typisierung.
    """
    cfg = _resolve_paths_inplace(cfg)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    clean_dir = Path(cfg["paths"]["clean_dir"])
    clean_dir.mkdir(parents=True, exist_ok=True)
    output_path = clean_dir / "events_master.csv"

    print(f"[Ingest] Using RAW dir: {raw_dir}")
    print(f"[Ingest] Clean output : {clean_dir}")

    # --- alle Maildateien suchen ---
    # --- alle Maildateien suchen ---
def _is_mail_file(p: Path):
    """Filtert echte Text-Mails heraus (.mbox, ohne Endung, keine versteckten Dateien)."""
    if p.name.startswith("."):  # versteckte Dateien (z. B. .DS_Store)
        return False
    if p.suffix.lower() in {".mbox", ".eml", ".txt"}:
        return True
    if p.suffix == "":  # dateien ohne Endung → typische Enron-Mails
        # schnelle Heuristik: enthalten sie "From:" oder "Subject:"?
        try:
            with open(p, "r", errors="ignore") as f:
                head = f.read(500)
            return ("From:" in head) and ("Subject:" in head)
        except Exception:
            return False
    return False

files = [p for p in raw_dir.rglob("*") if _is_mail_file(p)]
    files = [p for p in raw_dir.rglob("*") if p.is_file()]
    if sample_limit:
        files = files[:sample_limit]
    print(f"[Ingest] {len(files)} Dateien gefunden\n")

    # --- Container für Ergebnisse ---
    records = []

    for path in tqdm(files, desc="Parsing mails"):
        try:
            text = Path(path).read_text(errors="ignore")
            if not text.strip():
                continue

            # Header-Felder extrahieren
            sender = re.search(r"From:\s*(.*)", text)
            subject = re.search(r"Subject:\s*(.*)", text)
            date = re.search(r"Date:\s*(.*)", text)

            body = text.split("\n\n", 1)[-1].strip()
            msg_id = hashlib.md5(text.encode("utf-8")).hexdigest()
            thread_id = hashlib.md5((sender.group(1) if sender else "").encode()).hexdigest()

            record = {
                "event_id": msg_id,
                "thread_id": thread_id,
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

    # --- DataFrame erzeugen ---
    df = pd.DataFrame(records)
    if len(df) == 0:
        print("[Ingest] Keine Datensätze erzeugt.")
        df.to_csv(output_path, index=False)
        return output_path

    # --- Typisierung ergänzen (newsletter, stub, etc.) ---
    if flag_mail_types:
        df = flag_mail_types(df)
    else:
        df["content_type"] = "normal"

    # --- Analyse-Zulassung ---
    df["include_in_analysis"] = ~df["content_type"].isin(
        ["newsletter", "empty_or_stub", "attachment_dump"]
    )

    # --- Ausgabe ---
    df.to_csv(output_path, index=False)
    print(f"\n[Ingest] {len(df)} Datensätze → {output_path}")
    print("\n[content_type] Verteilung:\n", df["content_type"].value_counts())
    print("\n[include_in_analysis] True-Anteil:",
          (df["include_in_analysis"].mean() * 100).round(1), "%")

    return output_path