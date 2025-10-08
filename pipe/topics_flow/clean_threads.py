"""
pipe.clean_threads
==================
Bereinigt Event-Texte, erkennt Threads und filtert Mails nach Länge.
Erzeugt:
  - events_clean.csv            : alle Events inkl. Flags
  - events_roots_for_topics.csv : nur Thread-Roots mit <= MAX_CHARS

Beispiel:
---------
from pipe.setup_env import setup_environment
from pipe.clean_threads import clean_and_thread_events

env = setup_environment()
df_roots = clean_and_thread_events(env)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import Dict, Any

# ---------------------------------------------------------------------
# Parameter (können auch in Config stehen)
# ---------------------------------------------------------------------
THREAD_GAP_DAYS = 14
MAX_CHARS = 4000

CUT_MARKERS = [
    "-----Original Message-----",
    "Original Message",
    "Begin forwarded message:",
    "Start forwarded message",
    "Forwarded by",
    "Ursprüngliche Nachricht",
]
HEADER_PREFIXES = ("From:", "Sent:", "To:", "Cc:", "Bcc:", "Subject:", "Date:")
RE_MULTI_WS = re.compile(r"\s+")
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
RE_IMAGE = re.compile(r"\[(?:image|imag|inline image|cid:[^\]]*|logo|gif)\]", re.I)
SUBJ_RE_PREFIX = re.compile(r"(?i)^\s*((re|aw|wg|fwd?|tr|sv)\s*[:\]])\s*")
BRACKET_TAGS = re.compile(r"\[[^\]]*\]")


# ---------------------------------------------------------------------
# Cleaning-Funktionen
# ---------------------------------------------------------------------
def truncate_at_markers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    idxs = [text.find(m) for m in CUT_MARKERS if m in text]
    if idxs:
        cut = min(i for i in idxs if i >= 0)
        return text[:cut]
    return text


def strip_quote_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))


def strip_header_blocks(text: str) -> str:
    lines = text.splitlines()
    out, skip = [], False
    for line in lines:
        if any(line.lstrip().startswith(p) for p in HEADER_PREFIXES):
            skip = True
            continue
        if skip and line.strip() == "":
            skip = False
            continue
        if not skip:
            out.append(line)
    return "\n".join(out)


def clean_simple(s: str) -> str:
    s = s if isinstance(s, str) else ""
    s = truncate_at_markers(s)
    s = strip_quote_lines(s)
    s = strip_header_blocks(s)
    s = RE_IMAGE.sub(" ", s)
    s = RE_URL.sub(" ", s)
    s = RE_MULTI_WS.sub(" ", s).strip()
    return s


def norm_subject(s) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.strip()
    while SUBJ_RE_PREFIX.match(s):
        s = SUBJ_RE_PREFIX.sub("", s)
    s = BRACKET_TAGS.sub(" ", s)
    s = RE_MULTI_WS.sub(" ", s).strip().lower()
    return s or "(no-subject)"


def pick_root(g: pd.DataFrame) -> int:
    g2 = g.copy()
    if g2["date_parsed"].notna().any():
        g2 = g2.sort_values(["date_parsed", "event_id"], na_position="last")
    else:
        g2 = g2.sort_values("event_id")
    return int(g2.iloc[0]["event_id"])


# ---------------------------------------------------------------------
# Hauptpipeline
# ---------------------------------------------------------------------
def clean_and_thread_events(env: Dict[str, Any]) -> pd.DataFrame:
    clean_dir = Path(env["paths"]["clean_dir"])
    IN_CSV = clean_dir / "events_patched_with_participants.csv"
    OUT_ALL = clean_dir / "events_clean.csv"
    OUT_ROOT = clean_dir / "events_roots_for_topics.csv"

    assert IN_CSV.exists(), f"Input fehlt: {IN_CSV}"

    df = pd.read_csv(IN_CSV)

    # IDs und Basisfelder
    col_id = next((c for c in df.columns if c.lower() in {"event_id", "id", "eid"}), None)
    assert col_id, "Keine Event-ID-Spalte gefunden."
    df = df.rename(columns={col_id: "event_id"})

    col_subj = next((c for c in df.columns if c.lower() in {"subject", "subj"}), None)
    df["subject"] = df[col_subj] if col_subj else ""

    col_date = next((c for c in df.columns if c.lower() in {"date", "sent_at", "timestamp"}), None)
    df["date"] = df[col_date] if col_date else np.nan
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    # Cleaning
    df["text_full"] = df["text_full"].astype(str).fillna("")
    df["text_clean"] = df["text_full"].map(clean_simple)
    df["char_len"] = df["text_clean"].str.len()
    df["tok_len"] = df["text_clean"].str.findall(r"\w+").str.len()
    df["is_too_long"] = df["char_len"] > MAX_CHARS

    # Thread-Erkennung
    df["subject_norm"] = df["subject"].map(norm_subject)
    df = df.sort_values(["subject_norm", "date_parsed", "event_id"], na_position="last").reset_index(drop=True)

    gap = pd.Timedelta(days=THREAD_GAP_DAYS)
    thread_ids, current_tid, last_key, last_date = [], 0, None, None
    for _, r in df.iterrows():
        key, dt = r["subject_norm"], r["date_parsed"]
        if key != last_key:
            current_tid += 1
        elif pd.notna(dt) and pd.notna(last_date) and (dt - last_date) > gap:
            current_tid += 1
        thread_ids.append(current_tid)
        last_key, last_date = key, dt
    df["thread_id"] = thread_ids

    roots = df.groupby("thread_id", as_index=False).apply(lambda g: pd.Series({"root_event_id": pick_root(g)}))
    df = df.merge(roots, on="thread_id", how="left")
    df["is_thread_root"] = df["event_id"] == df["root_event_id"]

    # Outputs
    df.to_csv(OUT_ALL, index=False)
    roots_df = df[df["is_thread_root"] & (~df["is_too_long"])].copy()
    roots_df.to_csv(OUT_ROOT, index=False)

    print(f"[write] {OUT_ALL}  rows={len(df)}")
    print(f"[write] {OUT_ROOT}  roots={len(roots_df)} (<= {MAX_CHARS} Zeichen)")
    return roots_df


# ---------------------------------------------------------------------
if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment

    env = setup_environment()
    clean_and_thread_events(env)