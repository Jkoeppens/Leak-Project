"""
pipe.patch_events
=================
Liest events.csv und event_actor.csv, normalisiert E-Mails, vereinheitlicht Rollen
und erzeugt eine erweiterte events-Tabelle mit Teilnehmerlisten.
Ergebnis: events_patched_with_participants.csv im clean_dir.

Beispiel:
---------
from pipe.setup_env import setup_environment
from pipe.patch_events import patch_events_with_participants

env = setup_environment()
df = patch_events_with_participants(env)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re, json
from typing import Dict, Any


# ---------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------
def read_df_guessing(path: Path) -> pd.DataFrame:
    """Liest CSV mit Fallback-Encodings."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, engine="python")


def normalize_email(x: str) -> str | None:
    """Extrahiert und vereinheitlicht E-Mail-Adressen."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    m = re.search(r'<?([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})>?', s, re.I)
    if m:
        s = m.group(1)
    return s.strip("<>\"' ")


def split_multi(value):
    """Zerlegt Mehrfachangaben (Komma, Semikolon, Leerzeichen) in Listen."""
    if pd.isna(value):
        return []
    s = str(value).replace("\n", " ").replace("\r", " ")
    parts = re.split(r"[;, ]+", s)
    parts = [normalize_email(p) for p in parts if p and "@" in p]
    return [p for p in parts if p]


def agg_participants(df: pd.DataFrame) -> pd.Series:
    """Aggregiert Teilnehmer eines Events in Listen nach Rollen."""
    rec = {"FROM": [], "TO": [], "CC": [], "BCC": []}
    for _, row in df.iterrows():
        role = row["role_in_event"]
        email = row["email"]
        if role in rec:
            rec[role].append(email)
        elif role == "REPLY-TO":
            rec["TO"].append(email)
        else:
            rec["CC"].append(email)
    for k in rec:
        rec[k] = sorted(list(dict.fromkeys(rec[k])))
    return pd.Series(
        {
            "from_email": rec["FROM"][0] if rec["FROM"] else np.nan,
            "to_emails": rec["TO"],
            "cc_emails": rec["CC"],
            "bcc_emails": rec["BCC"],
            "n_participants": len(
                set(rec["FROM"] + rec["TO"] + rec["CC"] + rec["BCC"])
            ),
        }
    )


def to_json_list(x) -> str:
    """Sichere JSON-Serialisierung für Listen."""
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return "[]"


# ---------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------
def patch_events_with_participants(env: Dict[str, Any]) -> pd.DataFrame:
    """
    Kombiniert events.csv + event_actor.csv zu einer erweiterten Event-Tabelle.
    Gibt das DataFrame zurück und speichert es als CSV.
    """
    clean_dir = Path(env["paths"]["clean_dir"])
    EVENTS_CSV = clean_dir / "events.csv"
    ACTOR_CSV = clean_dir / "event_actor.csv"
    OUT_CSV = clean_dir / "events_patched_with_participants.csv"

    assert EVENTS_CSV.exists(), f"events.csv fehlt: {EVENTS_CSV}"
    assert ACTOR_CSV.exists(), f"event_actor.csv fehlt: {ACTOR_CSV}"

    # --- 1) Laden
    E_raw = read_df_guessing(EVENTS_CSV)
    A_raw = read_df_guessing(ACTOR_CSV)

    # --- 2) Spaltenheuristiken
    col_id = next((c for c in E_raw.columns if c.lower() in {"event_id", "id", "eid"}), None)
    col_text = next((c for c in E_raw.columns if c.lower() in {"text_full", "body", "text"}), None)
    col_subject = next((c for c in E_raw.columns if c.lower() in {"subject", "subj"}), None)
    col_date = next((c for c in E_raw.columns if c.lower() in {"date", "timestamp", "sent_at"}), None)
    assert col_id, f"Keine Event-ID-Spalte gefunden. Spalten: {list(E_raw.columns)}"

    if col_text is None:
        E_raw["text_full"] = E_raw.apply(
            lambda r: " ".join(str(v) for v in r.values if isinstance(v, str)), axis=1
        )
        col_text = "text_full"

    col_eid = next((c for c in A_raw.columns if c.lower() in {"event_id", "id", "eid"}), None)
    col_role = next((c for c in A_raw.columns if c.lower() in {"role_in_event", "role", "header"}), None)
    col_email = next((c for c in A_raw.columns if c.lower() in {"email", "address", "actor_email"}), None)
    assert all([col_eid, col_role, col_email]), \
        f"Erwarte event_actor-Spalten (event_id/role/email). Gefunden: {list(A_raw.columns)}"

    # --- 3) Normalisieren
    E = E_raw.rename(columns={col_id: "event_id", col_text: "text_full"})
    if col_subject:
        E = E.rename(columns={col_subject: "subject"})
    if col_date:
        E = E.rename(columns={col_date: "date"})

    A = A_raw.rename(columns={col_eid: "event_id", col_role: "role_in_event", col_email: "email"})
    A["email"] = A["email"].map(normalize_email)
    A = A[~A["email"].isna()]

    role_map = {"from": "FROM", "to": "TO", "cc": "CC", "bcc": "BCC", "sender": "FROM", "recipient": "TO"}
    A["role_in_event"] = (
        A["role_in_event"].astype(str).str.strip().str.lower().map(lambda r: role_map.get(r, r.upper()))
    )

    # --- 4) Aggregieren
    Agg = A.groupby("event_id", as_index=False).apply(agg_participants).reset_index(drop=True)

    # --- 5) Merge mit Events
    E2 = E.merge(Agg, how="left", on="event_id")

    for col in ("to", "cc", "bcc", "from"):
        if col in E_raw.columns and f"{col}_emails" not in E2.columns:
            if col == "from":
                E2["from_email"] = E2.get("from_email").fillna(E_raw[col].map(normalize_email))
            else:
                E2[f"{col}_emails"] = E_raw[col].map(split_multi)

    for listcol in ["to_emails", "cc_emails", "bcc_emails"]:
        if listcol not in E2.columns:
            E2[listcol] = [[] for _ in range(len(E2))]

    # --- 6) Qualitätsmetriken
    E2["text_len"] = E2["text_full"].astype(str).map(lambda s: len(re.findall(r"\w+", s)))
    E2["short_text_flag"] = E2["text_len"] < 12

    # --- 7) Speichern
    E2_out = E2.copy()
    for c in ["to_emails", "cc_emails", "bcc_emails"]:
        E2_out[c] = E2_out[c].map(to_json_list)

    E2_out.to_csv(OUT_CSV, index=False)
    print(f"[write] {OUT_CSV}  (rows={len(E2_out)})")

    print(E2_out[["event_id", "from_email", "n_participants", "text_len"]]
          .head(10).to_string(index=False))
    return E2_out


# ---------------------------------------------------------------------
if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment

    env = setup_environment()
    patch_events_with_participants(env)