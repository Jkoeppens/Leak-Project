# ============================================================
# pipe/audit/audit_ingest_types.py
# Audit: Typisierung & Integrität von Ingest-Outputs
# ============================================================

from pathlib import Path
import pandas as pd
from datetime import datetime

def audit_ingest_types(cfg, file_path=None, write_report=True):
    """
    Prüft content_type, include_in_analysis, parse_status
    und wichtige Spalten auf Vollständigkeit & Plausibilität.
    """
    clean_dir = Path(cfg["paths"]["clean_dir"])
    file_path = file_path or (clean_dir / "events_master.csv")
    df = pd.read_csv(file_path)

    print(f"[Audit] Prüfe {len(df)} Zeilen → {file_path.name}")

    # --- Grundlegende Kennzahlen ---
    total = len(df)
    parse_ok = (df["parse_status"] == "ok").sum()
    n_missing_sender = df["sender"].isna().sum()
    n_missing_subject = df["subject"].isna().sum()

    # --- Typenverteilung ---
    type_counts = df["content_type"].value_counts(dropna=False).to_dict()
    include_ratio = df["include_in_analysis"].mean() if "include_in_analysis" in df else None

    # --- Zeitprüfung ---
    ts_ok = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    ts_missing = ts_ok.isna().sum()
    ts_min = ts_ok.min()
    ts_max = ts_ok.max()

    # --- Ergebnisbericht ---
    summary = {
        "file": str(file_path),
        "rows": total,
        "parse_ok": int(parse_ok),
        "parse_failed": total - int(parse_ok),
        "missing_sender": int(n_missing_sender),
        "missing_subject": int(n_missing_subject),
        "timestamp_missing": int(ts_missing),
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
        "content_types": type_counts,
        "include_ratio": include_ratio,
        "run_timestamp": datetime.now().isoformat(),
    }

    # --- Markdown-Report schreiben ---
    if write_report:
        md_path = clean_dir / f"audit_ingest_types_{datetime.now():%Y-%m-%d_%H-%M}.md"
        with open(md_path, "w") as f:
            f.write("# Audit Report: Ingest Types\n\n")
            for k, v in summary.items():
                f.write(f"- **{k}**: {v}\n")
        print(f"[Audit] Markdown: {md_path.name}")

    # --- CSV-Zusammenfassung (für pipeline_audit.csv) ---
    csv_path = clean_dir / "audit_ingest_summary.csv"
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    print(f"[Audit] CSV: {csv_path.name}")

    return summary