# pipe/dynamic_levels.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Dict, Any
import json
import numpy as np
import pandas as pd

from .io import load_levels  # nutzt dein robustes IO

# ---------------- Heuristiken ----------------
def _coverage_topk(counts: pd.Series, target: float = 0.8) -> list[int]:
    v = counts.sort_values(ascending=False)
    if v.sum() == 0:
        return []
    c = (v.cumsum() / v.sum()).values
    k = int(np.searchsorted(c, target) + 1)
    return list(v.index[:k])

def _pick_min_size(LV: str, sizes: pd.Series, df_stats: pd.DataFrame) -> int:
    # Basis: xmin aus C (cluster_stats_summary.csv), Fallback 5, gedeckelt durch p95
    ms = 5
    if not df_stats.empty and LV in df_stats["level"].values:
        try:
            ms = int(df_stats.loc[df_stats["level"] == LV, "xmin"].iloc[0])
        except Exception:
            ms = 5
    ms = max(ms, 5)
    p95 = int(np.percentile(sizes.values, 95)) if len(sizes) else ms
    return int(min(ms, max(3, p95)))

def _pick_coverage_target(LV: str, sizes: pd.Series, df_stats: pd.DataFrame) -> float:
    # Aus Elbow-K via C eine Coverage ableiten; fallback 0.80
    if not df_stats.empty and LV in df_stats["level"].values:
        try:
            k_elbow = int(df_stats.loc[df_stats["level"] == LV, "kelbow"].iloc[0])
            v = sizes.sort_values(ascending=False).to_numpy()
            if len(v) and 1 <= k_elbow <= len(v):
                cov = float(v[:k_elbow].sum() / v.sum())
                return float(np.clip(cov, 0.60, 0.90))
        except Exception:
            pass
    return 0.80

def _build_parent_child(H: pd.DataFrame, parent_lv: str, child_lv: str, cap_children: int | None) -> dict[int, list[int]]:
    grp = H.groupby([parent_lv, child_lv]).size().reset_index(name="n").dropna()
    mp: dict[int, list[int]] = {}
    for p, sub in grp.groupby(parent_lv):
        kids = sub.sort_values("n", ascending=False)[child_lv].astype(int).tolist()
        if cap_children is not None:
            kids = kids[:cap_children]
        mp[int(p)] = kids
    return mp

def _pick_children_cap(df_parent_child_stats: pd.DataFrame | None) -> int:
    # Deckel via p95(n_children) im Bereich [10, 30], Fallback 30
    if df_parent_child_stats is None or "n_children" not in df_parent_child_stats.columns or len(df_parent_child_stats) == 0:
        return 30
    p95 = int(np.percentile(df_parent_child_stats["n_children"], 95))
    return int(min(max(p95, 10), 30))

# ---------------- Hauptfunktion ----------------
def build_dynamic_selection(cfg: dict, plot_levels: Optional[Iterable[str]] = None) -> dict[str, Any]:
    """
    Baut eine datengetriebene Auswahl pro Level_*:
    - min_size: aus cluster_stats_summary.csv (xmin) mit p95-Deckel
    - coverage: aus Elbow-K in Coverage umgerechnet (Fallback 0.80)
    - keep: Top-k Cluster-IDs nach Coverage
    - children: Parent->Child-Mapping, ggf. mit Cap

    Schreibt selection_dynamic.json und selection_meta.json nach outputs.org_dir.
    Returns: selection (dict)
    """
    # --- Daten & Pfade
    levels_csv = cfg["paths"]["levels"]
    cutdir = Path(cfg["outputs"]["cutoffs_dir"])
    orgdir = Path(cfg["outputs"]["org_dir"])
    orgdir.mkdir(parents=True, exist_ok=True)

    # Levels laden
    H = load_levels(levels_csv)
    level_cols = sorted([c for c in H.columns if str(c).startswith("level_")],
                        key=lambda s: int(s.split("_")[1]))

    # Stats aus C (optional)
    csum = cutdir / "cluster_stats_summary.csv"
    df_stats = pd.read_csv(csum) if csum.exists() else pd.DataFrame(columns=["level", "xmin", "kelbow", "k80", "k90"])

    # Parent->Child-Statistiken (optional eingespielt; wenn nicht vorhanden, bauen wir sie ad-hoc)
    parent_child_stats: dict[str, pd.DataFrame] = {}
    for i in range(len(level_cols) - 1):
        P, C = level_cols[i], level_cols[i + 1]
        pc_path = cutdir / f"parent_child_{P}_to_{C}.csv"
        parent_child_stats[C] = pd.read_csv(pc_path) if pc_path.exists() else None

    # Auswahl berechnen
    selection: dict[str, Any] = {"levels": level_cols, "keep": {}, "children": {}, "meta": {}}

    for i, LV in enumerate(level_cols):
        sizes = H[LV].value_counts().dropna().astype(int)

        # 1) min_size
        ms = _pick_min_size(LV, sizes, df_stats)
        sizes_f = sizes[sizes >= ms]

        # 2) coverage
        cov = _pick_coverage_target(LV, sizes_f, df_stats)

        # 3) keep-IDs
        keep_ids = _coverage_topk(sizes_f, cov)
        selection["keep"][LV] = list(map(int, keep_ids))
        selection["meta"][LV] = {
            "min_size": int(ms),
            "coverage": float(round(cov, 3)),
            "kept": int(len(keep_ids)),
            "clusters_total": int(sizes.size),
        }

        # 4) Kinder-Links
        if i < len(level_cols) - 1:
            P, C = LV, level_cols[i + 1]
            cap = _pick_children_cap(parent_child_stats.get(C))
            selection["children"][C] = _build_parent_child(H[H[P].isin(keep_ids)], P, C, cap_children=cap)
            selection["meta"][C] = {**selection["meta"].get(C, {}), "max_children_per_parent": int(cap)}

    # Speichern
    (orgdir / "selection_dynamic.json").write_text(json.dumps(selection, indent=2))
    (orgdir / "selection_meta.json").write_text(json.dumps(selection["meta"], indent=2))

    return selection
