# pipe/inspect_levels.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any
import numpy as np
import pandas as pd

from .io import load_levels  # nutzt dein robustes Level-Loading

# ---- kleine Helfer ----
def _coverage_k(counts: pd.Series, target: float = 0.8) -> int:
    v = counts.sort_values(ascending=False).to_numpy()
    if v.size == 0 or v.sum() == 0:
        return 0
    c = np.cumsum(v) / v.sum()
    return int(np.searchsorted(c, target) + 1)

def _describe_counts(counts: pd.Series) -> Dict[str, Any]:
    v = counts.to_numpy()
    P = (lambda q: float(np.percentile(v, q))) if v.size else (lambda q: float("nan"))
    return {
        "n_clusters": int(len(counts)),
        "min": int(counts.min()) if len(counts) else 0,
        "q25": P(25), "median": P(50), "q75": P(75),
        "p90": P(90), "p95": P(95), "p99": P(99),
        "max": int(counts.max()) if len(counts) else 0,
        "mean": float(counts.mean()) if len(counts) else 0.0,
        "std": float(counts.std(ddof=1)) if len(counts) > 1 else 0.0,
    }

def _level_cols(H: pd.DataFrame) -> list[str]:
    return sorted([c for c in H.columns if str(c).startswith("level_")],
                  key=lambda s: int(s.split("_")[1]))

# ---- 1) Clustergrößen + Summary schreiben ----
def write_cluster_size_reports(H: pd.DataFrame, outdir: str) -> pd.DataFrame:
    """
    Schreibt pro Level die Clustergrößen (CSV) und eine zusammenfassende Tabelle.
    Returns: summary-DataFrame
    """
    OUTDIR = Path(outdir); OUTDIR.mkdir(parents=True, exist_ok=True)
    levels = _level_cols(H)
    summary_rows = []

    for LV in levels:
        counts = H[LV].value_counts(dropna=True).astype(int).sort_values(ascending=False)
        stats  = _describe_counts(counts)
        k50 = _coverage_k(counts, 0.50)
        k80 = _coverage_k(counts, 0.80)
        k90 = _coverage_k(counts, 0.90)
        k95 = _coverage_k(counts, 0.95)

        # einzelne CSV
        counts.to_csv(OUTDIR / f"cluster_sizes_{LV}.csv", header=["size"])

        summary_rows.append({
            "level": LV, **stats,
            "k50": k50, "k80": k80, "k90": k90, "k95": k95,
            "ge_2": int((counts>=2).sum()),
            "ge_3": int((counts>=3).sum()),
            "ge_5": int((counts>=5).sum()),
            "ge_10": int((counts>=10).sum()),
            "sum_ge_5": int(counts[counts>=5].sum()),
        })

    df_summary = pd.DataFrame(summary_rows).sort_values("level")
    (OUTDIR / "cluster_size_summary_all_levels.csv").write_text(df_summary.to_csv(index=False))
    return df_summary

# ---- 2) Parent → Child Analyse ----
def write_parent_child_reports(H: pd.DataFrame, outdir: str) -> pd.DataFrame:
    """
    Schreibt Parent→Child-Tabellen für aufeinanderfolgende Levelpaare und
    legt eine Übersicht (parent_child_overview.csv) ab.
    Returns: overview-DataFrame
    """
    OUTDIR = Path(outdir); OUTDIR.mkdir(parents=True, exist_ok=True)
    levels = _level_cols(H)
    parent_child_overview = []

    for i in range(1, len(levels)):
        P_lv, C_lv = levels[i-1], levels[i]

        rows = []
        parents = H[P_lv].dropna().astype(int).unique().tolist()
        for pid in parents:
            members = H.loc[H[P_lv] == pid, ["node", C_lv]]
            kids = members[C_lv].dropna().astype(int)
            if kids.empty:
                rows.append({"parent": pid, "n_children": 0, "child_sizes_sum": 0, "top_children": ""})
                continue
            kid_counts = kids.value_counts().sort_values(ascending=False)
            top_children = ", ".join([f"{cid}:{sz}" for cid, sz in kid_counts.head(5).items()])
            rows.append({
                "parent": pid,
                "n_children": int(len(kid_counts)),
                "child_sizes_sum": int(kid_counts.sum()),
                "top_children": top_children,
            })

        df_pc = pd.DataFrame(rows).sort_values("n_children", ascending=False)
        df_pc.to_csv(OUTDIR / f"parent_child_{P_lv}_to_{C_lv}.csv", index=False)

        over = {
            "pair": f"{P_lv}->{C_lv}",
            "parents": int(len(df_pc)),
            "parents_with_children": int((df_pc["n_children"] > 0).sum()),
            "median_children_per_parent": float(df_pc["n_children"].median()) if len(df_pc) else 0.0,
            "p90_children_per_parent": float(np.percentile(df_pc["n_children"], 90)) if len(df_pc) else 0.0,
        }
        parent_child_overview.append(over)

    df_over = pd.DataFrame(parent_child_overview)
    (OUTDIR / "parent_child_overview.csv").write_text(df_over.to_csv(index=False))
    return df_over

# ---- 3) Simulation: min_size × coverage → #kept ----
def write_selection_simulation_grid(H: pd.DataFrame, outdir: str,
                                    min_sizes: Iterable[int] = (2,3,5,7,10,15,20),
                                    coverages: Iterable[float] = (0.5,0.7,0.8,0.9,0.95)) -> pd.DataFrame:
    """
    Erzeugt ein Gitter verschiedener min_size/coverage-Kombinationen und schreibt CSV.
    Returns: DataFrame mit 'level','min_size','coverage','clusters_kept','covered_persons'
    """
    OUTDIR = Path(outdir); OUTDIR.mkdir(parents=True, exist_ok=True)
    levels = _level_cols(H)

    sim_rows = []
    for LV in levels:
        counts = H[LV].value_counts(dropna=True).astype(int).sort_values(ascending=False)
        if counts.empty:
            continue
        for ms in min_sizes:
            cf = counts[counts >= int(ms)]
            total = int(cf.sum())
            ncl = int(len(cf))
            for cov in coverages:
                if total == 0:
                    k = 0
                else:
                    k = int(np.searchsorted(np.cumsum(cf.values) / total, float(cov)) + 1)
                sim_rows.append({
                    "level": LV,
                    "min_size": int(ms),
                    "coverage": float(cov),
                    "clusters_kept": int(min(k, ncl)),
                    "covered_persons": total,
                })

    sim = pd.DataFrame(sim_rows)
    (OUTDIR / "selection_simulation_grid.csv").write_text(sim.to_csv(index=False))
    return sim

# ---- 4) High-level Convenience ----
def inspect_levels_and_write(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lädt Levels aus cfg['paths']['levels'], schreibt alle Reports in cfg['outputs']['cutoffs_dir'].
    Returns: (df_summary, df_parent_child_overview, df_sim_grid)
    """
    H = load_levels(cfg["paths"]["levels"])
    # 'node' Spalte ist durch load_levels bereits normalisiert
    outdir = cfg["outputs"]["cutoffs_dir"]

    df_summary = write_cluster_size_reports(H, outdir)
    df_pc_over = write_parent_child_reports(H, outdir)
    df_sim = write_selection_simulation_grid(H, outdir)
    return df_summary, df_pc_over, df_sim
