# ============================================================
# pipe/ingest/flag_mail_types.py
# Typisierung von Mails: Newsletter, Threads, Leermails etc.
# ============================================================

import re

def flag_mail_types(df):
    """
    Fügt Spalte 'content_type' hinzu (newsletter, thread_history, stub, normal, attachment_dump).
    Diese Klassifikation dient zur Auditierung und Steuerung der Analysezulassung.
    """
    def classify(row):
        text = str(row.get("body_text", "")).lower()
        subj = str(row.get("subject", "")).lower()
        sender = str(row.get("sender", "")).lower()

        if len(text.strip()) == 0 or row.get("text_length", 0) < 10:
            return "empty_or_stub"
        if "unsubscribe" in text or "eyeforenergy" in sender or "newsletter" in subj:
            return "newsletter"
        if "begin 644" in text or "base64" in text:
            return "attachment_dump"
        if "-----original message-----" in text or subj.startswith("re:"):
            return "thread_history"
        return "normal"

    df["content_type"] = df.apply(classify, axis=1)
    return df