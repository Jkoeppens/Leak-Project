# ============================================================
# pipe/audit/audit_ingest.py
# Leak-Project – Audit + Diagnose für Ingest-Core
# Branch: consolidate/ingest-core
# ============================================================

import sys
from pathlib import Path
from datetime import datetime
from string import Template
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 0️⃣ Repo-Pfad hinzufügen (für Colab oder lokale Läufe)
# ------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

# ------------------------------------------------------------
# 1️⃣ Import interne Module
# ------------------------------------------------------------
from config.config import load_config
from pipe.ingest.ingest_core import ingest_core

try:
    from pipe.ingest.flag_mail_types import flag_mail_types
except ImportError:
    flag_mail_types = None

# ------------------------------------------------------------
# 2️⃣ Config laden + Platzhalter {root} auflösen
# ------------------------------------------------------------
cfg = load_config()

root = cfg["paths"]["root"]
for sec in ["paths", "outputs"]:
    if sec in cfg:
        for k, v in cfg[sec].items():
            if isinstance(v, str) and "{root}" in v:
                cfg[sec][k] = Template(v).safe_substitute(root=root)

print("\n[CONFIG CHECK]")
for k in ("raw_dir", "clean_dir"):
    print(f"{k:12s}: {cfg['paths'].get(k)}")

# ------------------------------------------------------------
# 3️⃣ Ingest starten
# ------------------------------------------------------------
print("\n[STEP] Ingest-Lauf startet …")
output_path = ingest_core(cfg, sample_limit=None)

CLEAN_DIR = Path(cfg["paths"]["clean_dir"])
output_path = Path(output_path)

# ------------------------------------------------------------
# 4️⃣ Daten laden
# ------------------------------------------------------------
try:
    df = pd.read_csv(output_path, dtype=str)
except Exception as e:
    print(f"[ERROR] Konnte {output_path} nicht laden → {e}")
    raise SystemExit(1)

df["text_length"] = pd.to_numeric(df.get("text_length"), errors="coerce")
df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce", utc=True)

# ------------------------------------------------------------
# 5️⃣ Mail-Typisierung (optional)
# ------------------------------------------------------------
if flag_mail_types:
    print("[STEP] Mail-Typisierung aktiv …")
    df = flag_mail_types(df)
else:
    df["content_type"] = "normal"

df["include_in_analysis"] = ~df["content_type"].isin(
    ["newsletter", "empty_or_stub", "attachment_dump"]
)

# ------------------------------------------------------------
# 6️⃣ Audit-Kennzahlen
# ------------------------------------------------------------
ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
length_stats = df["text_length"].describe(percentiles=[0.5, 0.9, 0.99]).to_dict()

summary = {
    "file": str(output_path),
    "rows": len(df),
    "parse_ok": (df["parse_status"] == "ok").sum(),
    "parse_failed": (df["parse_status"] != "ok").sum(),
    "missing_sender": df["sender"].isna().sum(),
    "missing_subject": df["subject"].isna().sum(),
    "missing_body": (df["body_text"].isna() | (df["body_text"].str.len() < 5)).sum(),
    "timestamp_missing": df["timestamp"].isna().sum(),
    "timestamp_min": ts_min,
    "timestamp_max": ts_max,
    "length_mean": float(length_stats.get("mean", 0)),
    "length_median": float(length_stats.get("50%", 0)),
    "length_max": float(length_stats.get("max", 0)),
    "include_ratio": float(df["include_in_analysis"].mean()),
    "run_timestamp": datetime.now().isoformat(),
}

# ------------------------------------------------------------
# 7️⃣ Reports erzeugen
# ------------------------------------------------------------
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")
summary_df = pd.DataFrame([summary])

md_path = CLEAN_DIR / f"audit_ingest_{TIMESTAMP}.md"
csv_path = CLEAN_DIR / "audit_ingest_summary.csv"

with open(md_path, "w") as f:
    f.write(f"# 📊 Audit Report – Ingest-Core ({TIMESTAMP})\n\n")
    for k, v in summary.items():
        f.write(f"- **{k}**: {v}\n")
    f.write("\n---\n## content_type-Verteilung\n")
    f.write(df["content_type"].value_counts().to_markdown())
    f.write("\n\n---\n## Top-Absender\n")
    f.write(df["sender"].value_counts().head(10).to_markdown())

summary_df.to_csv(csv_path, index=False)

# ------------------------------------------------------------
# 8️⃣ Konsolenausgabe
# ------------------------------------------------------------
print(f"\n[✅ Audit abgeschlossen]")
print(f"Markdown-Report: {md_path}")
print(f"CSV-Summary    : {csv_path}")
print("\n[Result Summary]")
for k, v in summary.items():
    print(f"{k:22s}: {v}")