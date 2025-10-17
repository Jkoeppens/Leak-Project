# ============================================================
# pipe/ingest/flag_mail_types.py
# Typisierung der Mails (Newsletter, Thread, Stub, Long, etc.)
# ============================================================

import pandas as pd
import numpy as np
import re


def flag_mail_types(df, cfg=None):
    """
    Typisiert Mails anhand einfacher Heuristiken:
    - empty_or_stub
    - newsletter
    - thread_history
    - long_mail (über Config steuerbar)
    """
    df = df.copy()
    df["content_type"] = "normal"

    # --- Threshold aus Config ---
    if cfg and "thresholds" in cfg and "long_mail" in cfg["thresholds"]:
        LONG_THRESHOLD = int(cfg["thresholds"]["long_mail"])
    else:
        LONG_THRESHOLD = 100_000  # Fallback

    # === Regeln ===
    df.loc[df["body_text"].isna() | (df["text_length"] < 20), "content_type"] = "empty_or_stub"
    df.loc[
        df["subject"].fillna("").str.contains("unsubscribe|newsletter|promotion|mailing list", case=False)
        | df["body_text"].fillna("").str.contains("unsubscribe|newsletter|promotion|mailing list", case=False),
        "content_type"
    ] = "newsletter"

    df.loc[
        df["body_text"].fillna("").str.contains("-----Original Message-----|Forwarded by|From:|Sent:", case=False),
        "content_type"
    ] = "thread_history"

    # --- Long Mail ---
    df.loc[df["text_length"] > LONG_THRESHOLD, "content_type"] = "long_mail"

    # --- Include-Flag ---
    df["include_in_analysis"] = ~df["content_type"].isin(
        ["newsletter", "empty_or_stub", "long_mail"]
    )

    return df