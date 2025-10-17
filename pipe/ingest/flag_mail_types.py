# ============================================================
# pipe/ingest/flag_mail_types.py
# Heuristische Klassifikation von Mailtypen (für Leak-Project)
# ============================================================

import pandas as pd
import numpy as np

def flag_mail_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Markiert Mails mit Heuristiken:
    - 'newsletter': lang oder enthält unsubscribe/HTML
    - 'empty_or_stub': extrem kurz
    - 'thread_history': enthält 'FW:' oder 'RE:'
    """
    df = df.copy()
    df["content_type"] = "normal"

    # 1️⃣ Leere oder extrem kurze Mails
    df.loc[df["text_length"].fillna(0) < 20, "content_type"] = "empty_or_stub"

    # 2️⃣ Newsletter / Massensendungen
    df.loc[
        (df["text_length"].fillna(0) > 8000)
        | (df["body_text"].str.contains("unsubscribe|<html|</body>", case=False, na=False)),
        "content_type"
    ] = "newsletter"

    # 3️⃣ Thread-Historien
    df.loc[
        df["subject"].str.contains(r"FW:|RE:", case=False, na=False),
        "content_type"
    ] = "thread_history"

    # 4️⃣ Analyse-Zulassung
    df["include_in_analysis"] = ~df["content_type"].isin(
        ["newsletter", "empty_or_stub", "attachment_dump"]
    )

    return df