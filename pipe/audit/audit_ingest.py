# ============================================================
# pipe/audit/audit_ingest.py
# Robuster Ingest-Audit mit Typkorrektur und Markdown-Report
# ============================================================

from pathlib import Path
from datetime import datetime
import pandas as pd
from pipe.ingest.ingest_core import ingest_core
from config.config import load_config


# ============================================================
# 1️⃣ Hauptfunktion: audit_ingest
# ============================================================
def audit_ingest(cfg=None, per_owner_limit=500, max_owners=3):
    """
    Führt den Ingest durch, prüft und protokolliert zentrale Qualitätsmetriken.
    Erstellt CSV- und Markdown-Report im Clean-Verzeichnis.
    """
    if cfg is None:
        cfg = load_config()

    print("[CONFIG CHECK]")
    print("raw_dir     :", cfg["paths"]["raw_dir"])
    print("clean_dir   :", cfg["paths"]["clean_dir"])
    print("\n[STEP] Ingest-Lauf startet …")

    # --- Ingest ausführen ---
    output_path = ingest_core(cfg, per_owner_limit=per_owner_limit, max_owners=max_owners)

    # --- CSV laden ---
    df = pd.read_csv(output_path)
    print(f"[LOAD] {len(df)} Zeilen aus {output_path}")

    # ------------------------------------------------------------
    # 🔧 2️⃣ Typkorrekturen (timestamp & text_length)
    # ------------------------------------------------------------
    if "timestamp" in df:
        # alles zu String wandeln, um float NaN loszuwerden
        df["timestamp"] = df["timestamp"].astype(str)

        # leere / NaN / None Werte korrigieren
        df["timestamp"] = df["timestamp"].replace(
            {"nan": None, "NaN": None, "": None, "None": None}
        )

        # konvertieren in UTC-Datetimes
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

        print(f"[TYPE] timestamp dtype: {df['timestamp'].dtype}, NaT: {df['timestamp'].isna().sum()}")

    if "text_length" in df:
        df["text_length"] = pd.to_numeric(df["text_length"], errors="coerce")
        print(f"[TYPE] text_length dtype: {df['text_length'].dtype}")

    # ------------------------------------------------------------
    # 🔍 3️⃣ Zusammenfassung berechnen
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
        "include_ratio": df["include_in_analysis"].mean() if "include_in_analysis" in df else None,
        "run_timestamp": datetime.now().isoformat()
    }

    # ------------------------------------------------------------
    # 🗂️ 4️⃣ Reports schreiben
    # ------------------------------------------------------------
    audit_dir = Path(cfg["paths"]["clean_dir"])
    audit_dir.mkdir(parents=True, exist_ok=True)

    audit_path_md = audit_dir / f"audit_ingest_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.md"
    audit_path_csv = audit_dir / "audit_ingest_summary.csv"

    pd.DataFrame([summary]).to_csv(audit_path_csv, index=False)

    with open(audit_path_md, "w", encoding="utf-8") as f:
        f.write("# 📋 Audit Ingest Report\n\n")
        for k, v in summary.items():
            f.write(f"- **{k}**: {v}\n")

    # ------------------------------------------------------------
    # 🧾 5️⃣ Ausgabe an Konsole
    # ------------------------------------------------------------
    print("\n[✅ Audit abgeschlossen]")
    print("Markdown-Report:", audit_path_md)
    print("CSV-Summary    :", audit_path_csv)
    print("\n[Result Summary]")
    for k, v in summary.items():
        print(f"{k:<22}: {v}")

    return summary


# ============================================================
# 2️⃣ CLI-kompatibler Startpunkt (optional)
# ============================================================
if __name__ == "__main__":
    cfg = load_config()
    audit_ingest(cfg)