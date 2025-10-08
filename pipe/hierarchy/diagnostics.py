from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

def outside_world_stats(E: pd.DataFrame, ow_label: str = "Outside world") -> dict:
    """
    Diagnose zur Sammelknoten-Logik 'Outside world'.

    Args:
        E: Edgelist-DataFrame mit Spalten ['src','dst','weight'].
        ow_label: Bezeichner des Outside-World-Knotens.

    Returns:
        dict mit:
            present (bool), edge_count (int), weight_sum (float),
            internal_edges (int), total_edges (int), internal_ratio (float)
    """
    for col in ("src", "dst", "weight"):
        if col not in E.columns:
            raise ValueError(f"Spalte '{col}' fehlt in E")

    mask_ow = (E["src"] == ow_label) | (E["dst"] == ow_label)
    present = bool(mask_ow.any())
    edge_count = int(mask_ow.sum())
    weight_sum = float(E.loc[mask_ow, "weight"].sum())

    total_edges = int(len(E))
    internal_edges = int(len(E[~mask_ow]))
    internal_ratio = float(internal_edges / total_edges) if total_edges else 0.0

    return {
        "present": present,
        "edge_count": edge_count,
        "weight_sum": weight_sum,
        "internal_edges": internal_edges,
        "total_edges": total_edges,
        "internal_ratio": internal_ratio,
    }

def save_outside_world_report(stats: dict, cfg: dict, filename: str = "outside_world.json") -> str:
    """
    Speichert die Diagnose als JSON unter outputs.diagnostics_dir.
    """
    outdir = Path(cfg["outputs"]["diagnostics_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return str(path)
