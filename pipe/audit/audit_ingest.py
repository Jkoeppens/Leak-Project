# ============================================================
# pipe/audit/audit_ingest.py
# Vollständiges, robustes Audit-Skript für Ingest-Pipeline
# ============================================================

from pathlib import Path
from datetime import datetime
import pandas as pd
from pipe.ingest.ingest_core import ingest_core
from config.config import load_config


def audit_ingest(cfg=None, per_owner_limit=500, max_owners=3):
    """
    Führt den Ingest-Prozess aus, prüft zentrale Kennzahlen
    und erzeugt einen Markdown- sowie CSV-Audit-Report.
    """
    # ------------------------------------------------------------
    # 1️⃣ Config laden
    # ------------------------------------------------------------
    if cfg is None:
        cfg = load_config()

    print("[CONFIG CHECK]")
    print("raw_dir     :", cfg["paths"]["raw_dir"])
    print("clean_dir   :", cfg["paths"]["clean_dir"])

    # ------------------------------------------------------------
    # 2️⃣ Ingest ausführen
    # ------------------------------------------------------------
    print("\n[STEP] Ingest-Lauf startet …")
    output_path = ingest_core(cfg, per_owner_limit=per_owner_limit, max_owners=max_owners)

    # ------------------------------------------------------------
    # 3️⃣ CSV laden und Typkorrekturen anwenden
    # ------------------------------------------------------------
    print("\n[STEP] Mail-Typisierung aktiv …")
    df = pd.read_csv(output_path)

    # --- Fix Timestamp & Textlänge ---
    if "timestamp" in df:
        df["timestamp"] = df["timestamp"].astype(str)
        df["timestamp"] = df["timestamp"].replace(
            {"nan": None, "NaN": None, "": None, "None": None}
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        print(f"[TYPE FIX] timestamp dtype: {df['timestamp'].dtype} | NaT count: {df['timestamp'].isna().sum()}")

    if "text_length" in df:
        df["text_length"] = pd.to_numeric(df["text_length"], errors="coerce")

    # ------------------------------------------------------------
    # 4️⃣ Audit-Metriken berechnen
    # ------------------------------------------------------------
    summary = {
        "file": str(output_path),
        "rows": len(df),
        "parse_ok": (df["parse_status"] == "ok").sum() if "parse_status" in df else len(df),
        "parse_failed": (df["parse_status"] != "ok").sum() if "parse_status" in df else 0,
        "missing_sender": df["sender"].isna().sum() if "sender" in df else None,
        "missing_subject": df["subject"].isna().sum() if "subject" in df else None,
        "missing_body": df["body_text"].isna().sum() if "body_text" in df else None,
        "timestamp_missing": df["timestamp"].isna().sum() if "timestamp" in df else None,
        "timestamp_min": df["timestamp"].min() if "timestamp" in df else None,
        "timestamp_max": df["timestamp"].max() if "timestamp" in df else None,
        "length_mean": df["text_length"].mean() if "text_length" in df else None,
        "length_median": df["text_length"].median() if "text_length" in df else None,
        "length_max": df["text_length"].max() if "text_length" in df else None,
        "include_ratio": (
            df["include_in_analysis"].mean()
            if "include_in_analysis" in df
            else None
        ),
        "run_timestamp": datetime.now().isoformat(),
    }

    # ------------------------------------------------------------
    # 5️⃣ Reports schreiben
    # ------------------------------------------------------------
    audit_path_md = Path(cfg["paths"]["clean_dir"]) / f"audit_ingest_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.md"
    audit_path_csv = Path(cfg["paths"]["clean_dir"]) / "audit_ingest_summary.csv"

    pd.DataFrame([summary]).to_csv(audit_path_csv, index=False)

    with open(audit_path_md, "w") as f:
        f.write("# Audit Ingest Report\n\n")
        for k, v in summary.items():
            f.write(f"- **{k}**: {v}\n")

    print("\n[✅ Audit abgeschlossen]")
    print("Markdown-Report:", audit_path_md)
    print("CSV-Summary    :", audit_path_csv)
    print("\n[Result Summary]")
    for k, v in summary.items():
        print(f"{k:25}: {v}")

    return summary


# ============================================================
# 6️⃣ Direktlauf (wenn Datei standalone ausgeführt wird)
# ============================================================
if __name__ == "__main__":
    cfg = load_config()
    audit_ingest(cfg, per_owner_limit=100, max_owners=3)