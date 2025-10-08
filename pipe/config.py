# pipe/config.py
from __future__ import annotations
from pathlib import Path
import yaml, json

def repo_root() -> Path:
    # Finde Git-Root; wenn nicht vorhanden, nimm zwei Ebenen über pipe/
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return parent
    return p.parents[2]  # .../pipe/config.py -> repo root

def _expand_placeholders(obj, root_str: str):
    if isinstance(obj, str):
        return obj.replace("{root}", root_str)
    if isinstance(obj, dict):
        return {k: _expand_placeholders(v, root_str) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_placeholders(v, root_str) for v in obj]
    return obj

def _abspath_paths(paths: dict, rr: Path) -> dict:
    out = {}
    for k, v in paths.items():
        if isinstance(v, str) and not Path(v).is_absolute():
            out[k] = str((rr / v).resolve())
        else:
            out[k] = v
    return out

def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config() -> dict:
    rr = repo_root()
    with open(rr / "config" / "default.yaml", "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    local_path = rr / "config" / "local.yaml"
    local = {}
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            local = yaml.safe_load(f)

    # Merge & expand
    merged = _deep_merge(base, local)
    root_str = str(rr) if merged.get("paths", {}).get("root", "{root}") == "{root}" else merged["paths"]["root"]
    merged = _expand_placeholders(merged, root_str)

    # Pfade absolutieren
    if "paths" in merged:
        merged["paths"] = _abspath_paths(merged["paths"], rr)
    if "outputs" in merged:
        merged["outputs"] = _abspath_paths(merged["outputs"], rr)

    return merged
