# ============================================================
# pipe/ingest/flag_mail_types.py
# Typisierung der Mails (Newsletter, Thread, Stub, Long, etc.)
# ============================================================

import pandas as pd
import numpy as np
import re


def flag_mail_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Typisiert Mails anhand einfacher Heuristiken:
      - empty_or_stub: sehr kurz oder leer
      - newsletter: enthält unsubscribe, promotion, listserv
      - thread_history: enthält Forward/Original-Message
      - long_mail: extrem lange Nachrichten (z. B. Dumps)
    """
    df = df.copy()

    # --- Basis: Alle "normal" setzen
    df["content_type"] = "normal"

    # --- 1️⃣ Empty or Stub
    df.loc[
        df["body_text"].isna() | (df["text_length"] < 20),
        "content_type"
    ] = "empty_or_stub"

    # --- 2️⃣ Newsletter
    newsletter_patterns = [
        r"unsubscribe", r"newsletter", r"mailing list", r"promotion", r"to stop receiving"
    ]
    df.loc[
        df["subject"].fillna("").str.contains("|".join(newsletter_patterns), case=False)
        | df["body_text"].fillna("").str.contains("|".join(newsletter_patterns), case=False),
        "content_type"
    ] = "newsletter"

    # --- 3️⃣ Thread History
    thread_markers = [
        r"-----Original Message-----",
        r"Forwarded by",
        r"From:",
        r"Sent:",
    ]
    df.loc[
        df["body_text"].fillna("").str.contains("|".join(thread_markers), case=False),
        "content_type"
    ] = "thread_history"

    # --- 4️⃣ Long Mail (NEU)
    LONG_THRESHOLD = 10_000
    df.loc[
        df["text_length"] > LONG_THRESHOLD,
        "content_type"
    ] = "long_mail"

    # --- 5️⃣ Include/Exclude Flag
    df["include_in_analysis"] = ~df["content_type"].isin([
        "newsletter", "empty_or_stub", "long_mail"
    ])

    return df