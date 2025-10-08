# pipe/signif_debug.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

def _find_signif_files(signif_dir: Path) -> Dict[str, Path]:
    """
    Sucht Signifikanz-CSV je Level. Akzeptiert:
      - cluster_significance_level_*.csv
      - cluster_significance_*.csv
    Liefert dict level_name -> Pfad.
    """
    files = {}
    # Variante 1 (empfohlen aus F): cluster_significance_level_X.csv
    for p in sorted(signif_dir.glob("cluster_significance_level_*.csv")):
        lv = p.stem.replace("cluster_significance_", "")
        files[lv] = p
    # Variante 2 (Fallback): cluster_significance_*.csv
    for p in sorted(signif_dir.glob("cluster_significance_*.csv")):
        lv = p.stem.replace("cluster_significance_", "")
        files.setdefault(lv, p)
    return files

def _robust_z(df: pd.DataFrame) -> pd.DataFrame:
    z = df["z_score"].astype(float)
    med = np.median(z)
    mad = np.median(np.abs(z - med))
    iqr = np.subtract(*np.percentile(z, [75, 25]))
    out = df.copy()
    out["z_mad"] = (z - med) / (mad if mad > 0 else np.nan) * 0.6745
    out["z_iqr"] = (z - med) / (iqr if iqr > 0 else np.nan) * 0.7413
    return out

def _moment_summary(df: pd.DataFrame, level: str) -> dict:
    z = df["z_score"].astype(float)
    mu = float(z.mean())
    sigma = float(z.std(ddof=1)) if len(z) > 1 else np.nan
    return {
        "level": level,
        "count": int(len(df)),
        "z_mu": mu,
        "z_sigma": sigma,
        "z_min": float(z.min()) if len(z) else np.nan,
        "z_p50": float(np.percentile(z, 50)) if len(z) else np.nan,
        "z_p90": float(np.percentile(z, 90)) if len(z) else np.nan,
        "z_max": float(z.max()) if len(z) else np.nan,
        "sigma_near_zero": bool(np.isfinite(sigma) and sigma < 1e-6),
        "n_nodes_min": int(df["n_nodes"].min()) if "n_nodes" in df and len(df) else 0,
        "n_nodes_p50": float(np.percentile(df["n_nodes"], 50)) if "n_nodes" in df and len(df) else 0.0,
        "n_nodes_max": int(df["n_nodes"].max()) if "n_nodes" in df and len(df) else 0,
    }

def _extremes_by_abs_z(df: pd.DataFrame, level: str, k: int = 10) -> pd.DataFrame:
    d = df.copy()
    d["abs_z"] = d["z_score"].abs()
    take = pd.concat([d.nlargest(k, "abs_z"), d.nsmallest(k, "abs_z")], ignore_index=True).drop_duplicates()
    keep_cols = ["level","cluster_id","n_nodes","m_in_observed","exp_m_in","var_m_in",
                 "density_observed","z_score","z_mad","z_iqr"]
    for c in keep_cols:
        if c not in take.columns:
            take[c] = np.nan
    take["level"] = level
    return take[keep_cols]

def _notes_for_level(df: pd.DataFrame, level: str) -> List[str]:
    notes: List[str] = []
    if "z_score" in df:
        z = df["z_score"].astype(float)
        sigma = float(z.std(ddof=1)) if len(z) > 1 else np.nan
        if np.isfinite(sigma) and sigma < 1e-6:
            notes.append(f"[{level}] σ≈0 → winzige Streuung; kleine Unterschiede blasen z auf.")
    if "var_m_in" in df and (df["var_m_in"] <= 1e-9).any():
        hit = df.loc[df["var_m_in"] <= 1e-9, ["cluster_id","n_nodes","m_in_observed","exp_m_in","var_m_in"]].head(5)
        notes.append(f"[{level}] var_m_in≈0 in {int((df['var_m_in']<=1e-9).sum())} Clustern → z→∞. Beispiele:\n{hit.to_string(index=False)}")
    if "density_observed" in df:
        d0 = int((df["density_observed"] == 0).sum())
        d1 = int((df["density_observed"] >= 0.9999).sum())
        if d0 or d1:
            notes.append(f"[{level}] Dichte=0: {d0} | Dichte≈1: {d1} → starke Ausreißer möglich.")
    return notes

def run_signif_debug(cfg: dict, topk: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Liest alle cluster_significance_*.csv, berechnet robuste Kennzahlen,
    schreibt z_debug_summary.csv und z_debug_outliers.csv in outputs.signif_dir.
    Returns: (summary_df, outliers_df, notes_str)
    """
    signif_dir = Path(cfg["outputs"]["signif_dir"])
    if not signif_dir.exists():
        raise FileNotFoundError(f"Significance-Ordner fehlt: {signif_dir}")

    files = _find_signif_files(signif_dir)
    if not files:
        raise FileNotFoundError(f"Keine cluster_significance_*.csv in {signif_dir} gefunden.")

    summaries = []
    outliers = []
    all_notes: List[str] = []

    for lv, p in files.items():
        df = pd.read_csv(p)
        if "z_score" not in df.columns:
            continue
        df = _robust_z(df)
        summaries.append(_moment_summary(df, lv))
        outliers.append(_extremes_by_abs_z(df, lv, k=topk))
        all_notes.extend(_notes_for_level(df, lv))

    summary_df = pd.DataFrame(summaries).sort_values("level")
    outliers_df = (pd.concat(outliers, ignore_index=True)
                     .sort_values(["level","z_score"], ascending=[True, False]))

    out_sum = signif_dir / "z_debug_summary.csv"
    out_ext = signif_dir / "z_debug_outliers.csv"
    out_sum.write_text(summary_df.to_csv(index=False))
    out_ext.write_text(outliers_df.to_csv(index=False))

    notes_str = "\n".join(all_notes) if all_notes else "[OK] Keine offensichtlichen numerischen Pathologien gefunden."
    return summary_df, outliers_df, notes_str
