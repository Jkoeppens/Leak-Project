# pipe/ingest/flag_mail_types.py
import pandas as pd

def flag_mail_types(df: pd.DataFrame) -> pd.DataFrame:
    df["content_type"] = "normal"
    df.loc[df["subject"].str.contains("unsubscribe|newsletter|offer", case=False, na=False), "content_type"] = "newsletter"
    df.loc[df["body_text"].str.match(r"^-----Original Message-----", na=False), "content_type"] = "thread_history"
    df.loc[df["text_length"] < 20, "content_type"] = "empty_or_stub"
    return df