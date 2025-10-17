# ============================================================
# pipe/audit/audit_ingest.py
# Laufende Diagnose + Audit für Ingest-Core
# ============================================================

from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from config.config import load_config
from pipe.ingest.ingest_core import ingest_core

# Optional: Mail-Typisierung laden (falls vorhanden)
try:
    from pipe.ingest.flag_mail_types import flag_mail_types
except ImportError:
    flag_mail_types = None

# ------------------------------------------------------------
# 1️⃣ Setup + Lauf
# ------------------------------------------------------------
cfg = load_config()
output_path = ingest_core(cfg, sample_limit=None)  # alle Dateien verarbeiten

CLEAN_DIR = Path(cfg["paths"]["clean_dir"])
df = pd.read_csv(output_path, dtype=str)
df["text_length"] = pd.to_numeric(df["text_length"], errors="coerce")

# ------------------------------------------------------------
# 2️⃣ Mail-Typisierung (optional)
# ------------------------------------------------------------
if flag_mail_types:
    print("[MailTypes] Typisierung aktiv ...")
    df = flag_mail_types(df)
else:
    print("[MailTypes] Kein Typisierungsmodul gefunden – alles 'normal'")
    df["content_type"] = "normal"

# Einfache Ableitung „include_in_analysis“
df["include_in_analysis"] = ~df["content_type"].isin(
    ["newsletter", "empty_or_stub", "attachment_dump"]
)

# ------------------------------------------------------------
# 3️⃣ Audit-Metriken
# ------------------------------------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

summary = {
    "file": str(output_path),
    "rows": len(df),
    "parse_ok": (df["parse_status"] == "ok").sum(),
    "parse_failed": (df["parse_status"] != "ok").sum(),
    "missing_sender": df["sender"].isna().sum(),
    "missing_subject": df["subject"].isna().sum(),
    "missing_body": (df["body_text"].isna() | (df["body_text"].str.len() < 5)).sum(),
    "timestamp_missing": df["timestamp"].isna().sum(),
    "timestamp_min": pd.to_datetime(df["timestamp"], errors="coerce").min(),
    "timestamp_max": pd.to_datetime(df["timestamp"], errors="coerce").max(),
    "length_mean": float(df["text_length"].mean()),
    "length_median": float(df["text_length"].median()),
    "length_max": float(df["text_length"].max()),
    "include_ratio": float(df["include_in_analysis"].mean()),
    "run_timestamp": datetime.now().isoformat(),
}

summary_df = pd.DataFrame([summary])
summary_csv = CLEAN_DIR / "audit_ingest_summary.csv"
summary_df.to_csv(summary_csv, index=False)

# ------------------------------------------------------------
# 4️⃣ Markdown-Report
# ------------------------------------------------------------
md_path = CLEAN_DIR / f"audit_ingest_{timestamp}.md"
with open(md_path, "w") as f:
    f.write(f"# Audit Report (Ingest)\n\n")
    for k, v in summary.items():
        f.write(f"- **{k}**: {v}\n")
    f.write("\n---\n## content_type-Verteilung\n")
    f.write(df["content_type"].value_counts().to_markdown())
    f.write("\n\n---\n## Top-Absender\n")
    f.write(df["sender"].value_counts().head(10).to_markdown())

print(f"\n[✅ Audit abgeschlossen]")
print(f"Markdown: {md_path}")
print(f"CSV:      {summary_csv}")