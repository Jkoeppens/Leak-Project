# pipe/ingest_any.py
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Tuple, Optional, List
import csv
import datetime as dt
import email
from email import policy
from email.utils import parsedate_to_datetime, getaddresses
import hashlib
import mailbox
import os
import re
import time
from collections import defaultdict
import pandas as pd

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def _norm_email(addr: str) -> str:
    if not addr:
        return ""
    m = EMAIL_RE.search(addr.lower())
    return m.group(0) if m else addr.lower().strip()

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def _parse_recipients(msg, field: str) -> list[str]:
    raw = msg.get_all(field, [])
    pairs = getaddresses(raw)
    out = []
    for _, a in pairs:
        na = _norm_email(a)
        if na:
            out.append(na)
    return out

def _message_ts(msg) -> Optional[dt.datetime]:
    d = msg.get("date")
    if not d:
        return None
    try:
        ts = parsedate_to_datetime(d)
        if ts is None:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        return ts.astimezone(dt.timezone.utc)
    except Exception:
        return None

def _extract_text(msg, keep: bool, max_chars: int) -> str:
    """Nur wenn keep=True. Versucht text/plain zu nehmen; sonst leer."""
    if not keep:
        return ""
    body_parts: list[str] = []
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body_parts.append(part.get_content())
                    except Exception:
                        payload = part.get_payload(decode=True)
                        if payload is None:
                            continue
                        body_parts.append(payload.decode(part.get_content_charset() or "utf-8", errors="ignore"))
        else:
            try:
                body_parts.append(msg.get_content())
            except Exception:
                payload = msg.get_payload(decode=True)
                if payload is not None:
                    body_parts.append(payload.decode(msg.get_content_charset() or "utf-8", errors="ignore"))
    except Exception:
        pass
    text = "\n".join(bp for bp in body_parts if bp)  # FIX: dein Paste hatte hier einen Zeilenbruch-Bug
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text

def _list_all_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            files.append(Path(dirpath) / fn)
    return files

def _iter_messages_from_dir(files: List[Path]) -> Iterator[tuple[object, Path]]:
    for fp in files:
        try:
            with open(fp, "rb") as f:
                data = f.read()
            msg = email.message_from_bytes(data, policy=policy.default)
            yield msg, fp
        except Exception:
            continue

def _iter_messages(path: Path, prelisted: Optional[List[Path]] = None) -> Iterator[tuple[object, Path]]:
    if path.is_dir():
        files = prelisted if prelisted is not None else _list_all_files(path)
        yield from _iter_messages_from_dir(files)
    elif str(path).lower().endswith(".mbox"):
        mbox = mailbox.mbox(str(path))
        for msg in mbox:
            yield msg, path
    elif str(path).lower().endswith(".eml"):
        with open(path, "rb") as f:
            data = f.read()
        msg = email.message_from_bytes(data, policy=policy.default)
        yield msg, path
    else:
        raise SystemExit(f"Unsupported path: {path}")

def ingest_emails(cfg: dict, *, input_root: Path | None = None) -> dict[str, str]:
    """
    Liest E-Mails aus input_root (dir/mbox/eml), schreibt:
      - {clean_dir}/events.csv
      - {clean_dir}/event_actor.csv
      - {paths.edges}
    Returns: dict mit geschriebenen Pfaden.
    """
    raw_dir = Path(cfg["paths"]["raw_dir"])
    sub = cfg.get("ingest", {}).get("input_subdir", "")
    in_root = Path(input_root) if input_root else (raw_dir / sub if sub else raw_dir)
    clean_dir = Path(cfg["paths"]["clean_dir"])
    out_edges = Path(cfg["paths"]["edges"])

    progress_every = int(cfg.get("ingest", {}).get("progress_every", 500))
    keep_body = bool(cfg.get("ingest", {}).get("keep_body_text", False))
    max_chars = int(cfg.get("ingest", {}).get("body_text_max_chars", 5000))

    t0 = time.perf_counter()
    files = _list_all_files(in_root) if in_root.is_dir() else [in_root]
    total = len(files)
    if total == 0:
        raise FileNotFoundError(f"Keine Dateien gefunden unter: {in_root}")

    clean_dir.mkdir(parents=True, exist_ok=True)
    out_edges.parent.mkdir(parents=True, exist_ok=True)

    events_path = clean_dir / "events.csv"
    actors_path = clean_dir / "event_actor.csv"
    edges_tmp = clean_dir / "edges_tmp.csv"  # für Aggregation
    edges_path = out_edges

    edge_counts: dict[tuple[str, str], list] = defaultdict(lambda: [0, None])  # weight, last_ts

    with open(events_path, "w", newline="", encoding="utf-8") as events_f, \
         open(actors_path, "w", newline="", encoding="utf-8") as actors_f:

        events_w = csv.writer(events_f)
        actors_w = csv.writer(actors_f)
        events_w.writerow(["event_id","event_type","ts_start","subject","body_text","thread_id","parent_id","system_source","sha256"])
        actors_w.writerow(["event_id","email","role_in_event"])

        n_msgs = 0
        n_errors = 0
        last_pct = -1

        for i, (msg, source_path) in enumerate(_iter_messages(in_root, prelisted=files)):
            try:
                n_msgs += 1
                eid = n_msgs
                subj = (msg.get("subject") or "").strip()
                ts = _message_ts(msg) or dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
                ts_iso = ts.astimezone(dt.timezone.utc).isoformat()

                frm = _norm_email(msg.get("from", ""))
                tos = _parse_recipients(msg, "to")
                ccs = _parse_recipients(msg, "cc")

                mid = (msg.get("message-id") or "").strip()
                in_reply_to = (msg.get("in-reply-to") or "").strip()

                body_text = _extract_text(msg, keep=keep_body, max_chars=max_chars)
                h = _sha256(f"{ts_iso}|{subj}|{frm}|{';'.join(sorted(tos))}")

                try:
                    system_source = os.path.relpath(str(source_path), start=str(in_root))
                except Exception:
                    system_source = os.path.basename(str(source_path))

                events_w.writerow([eid, "email", ts_iso, subj, body_text, mid, in_reply_to, system_source, h])

                if frm:
                    actors_w.writerow([eid, frm, "from"])
                for r in tos:
                    actors_w.writerow([eid, r, "to"])
                for r in ccs:
                    actors_w.writerow([eid, r, "cc"])

                recips = set(tos + ccs)
                for r in recips:
                    if not frm or not r or frm == r:
                        continue
                    key = (frm, r)
                    edge_counts[key][0] += 1
                    edge_counts[key][1] = ts_iso

            except Exception:
                n_errors += 1
                continue

            if total > 0:
                pct = int((i + 1) * 100 / total)
                if pct != last_pct and pct % 5 == 0:
                    print(f"    {pct}% ({i+1}/{total}) …")
                    last_pct = pct
            if n_msgs % progress_every == 0:
                print(f"    processed {n_msgs} messages …")

    # Aggregierte Kanten schreiben
    with open(edges_path, "w", newline="", encoding="utf-8") as edges_f:
        edges_w = csv.writer(edges_f)
        edges_w.writerow(["src","dst","weight","last_ts","channel"])
        for (s, d), (w, last_ts) in edge_counts.items():
            edges_w.writerow([s, d, w, last_ts, "email"])

    dt_sec = time.perf_counter() - t0
    print(f"Done. Processed messages: {n_msgs}, errors: {n_errors} | elapsed {dt_sec:.2f}s")
    print(f"Wrote: {events_path}")
    print(f"Wrote: {actors_path}")
    print(f"Wrote: {edges_path}")
    return {"events": str(events_path), "actors": str(actors_path), "edges": str(edges_path)}

# Optionaler CLI-Wrapper
def main_cli():
    import argparse, json
    parser = argparse.ArgumentParser(description="Ingest emails into clean/events & derived edges")
    parser.add_argument("--input", type=str, default=None, help="Pfad zu dir/.mbox/.eml; sonst cfg.paths.raw_dir + ingest.input_subdir")
    args = parser.parse_args()

    # lazy import, um harte Abhängigkeit zu vermeiden
    from .config import load_config
    cfg = load_config()
    res = ingest_emails(cfg, input_root=Path(args.input) if args.input else None)
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main_cli()
