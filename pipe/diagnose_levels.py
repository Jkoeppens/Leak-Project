# pipe/diagnose_levels.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw

def _coverage_topk(counts: pd.Series, target: float = 0.8) -> int:
    v = counts.sort_values(ascending=False).to_numpy()
    if v.size == 0 or v.sum() == 0:
        return 0
    c = np.cumsum(v) / v.sum()
    return int(np.searchsorted(c, target) + 1)

def _elbow_k(counts: pd.Series) -> int:
    v = counts.sort_values(ascending=False).to_numpy()
    if v.size < 3:
        return int(v.size)
    c = np.cumsum(v) / v.sum()
    curv = np.zeros_like(c)
    curv[1:-1] = c[:-2] - 2 * c[1:-1] + c[2:]
    return int(np.argmax(curv) + 1)

def _fit_powerlaw(values: np.ndarray) -> dict:
    try:
        fit = powerlaw.Fit(values, discrete=True, xmin=None, verbose=False)
        R, p = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
        return dict(
            xmin=int(fit.xmin),
            alpha=float(fit.alpha),
            ks_D=float(fit.D),
            lr_R=float(R),
            lr_p=float(p),
            _fit=fit,
        )
    except Exception:
        # Fallback auf Median als xmin, wenn Fit fehlschlägt
        xmin = int(np.percentile(values, 50)) if values.size else 5
        return dict(xmin=xmin, alpha=float("nan"), ks_D=float("nan"),
                    lr_R=float("nan"), lr_p=float("nan"), _fit=None)

def analyze_levels(
    H: pd.DataFrame,
    outdir: str,
    plot_levels: Optional[Iterable[str]] = ("level_2", "level_3"),
) -> pd.DataFrame:
    """
    Analyse je Level_*: Clustergrößen, Power-Law-Fit, Heuristiken (k80/k90/Elbow).
    Schreibt CSV + (optional) CCDF-Plots.

    Returns: DataFrame mit Metriken pro Level.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    level_cols = [c for c in H.columns if str(c).startswith("level_")]
    rows = []

    for LV in level_cols:
        counts = H[LV].value_counts(dropna=True).astype(int)
        if counts.empty:
            continue

        pl = _fit_powerlaw(counts.values)
        k80 = _coverage_topk(counts, 0.80)
        k90 = _coverage_topk(counts, 0.90)
        kel = _elbow_k(counts)
        min_size = max(int(pl["xmin"]), 5)

        # Plot nur, wenn gewünscht und Fit verfügbar
        if plot_levels and LV in set(plot_levels) and pl.get("_fit") is not None:
            fit = pl["_fit"]
            ax = fit.plot_ccdf(label="Daten")
            fit.power_law.plot_ccdf(ax=ax, linestyle="--",
                                    label=f"Power-law (α={fit.alpha:.2f}, xmin={int(fit.xmin)})")
            ax.set_xlabel("Clustergröße")
            ax.set_ylabel("CCDF")
            ax.set_title(f"CCDF & Power-law – {LV} (R={pl['lr_R']:.2f}, p={pl['lr_p']:.3f})")
            ax.legend()
            plt.tight_layout()
            plt.savefig(str(Path(outdir) / f"ccdf_{LV}.png"), dpi=140)
            plt.close()

        rows.append(dict(
            level=LV,
            n_clusters=int(len(counts)),
            n_people=int(len(H)),
            xmin=int(pl["xmin"]),
            alpha=pl["alpha"],
            ks_D=pl["ks_D"],
            lr_R=pl["lr_R"],
            lr_p=pl["lr_p"],
            k80=int(k80),
            k90=int(k90),
            kelbow=int(kel),
            min_size=int(min_size),
        ))

    df = pd.DataFrame(rows).sort_values("level")
    (Path(outdir) / "cluster_stats_summary.csv").write_text(df.to_csv(index=False))
    return df
