# pipe/signif.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from .io import load_levels  # nutzt dein robustes Level-Loading

# ---------- Helfer: Kanten normalisieren ----------
def _normalize_edges(E: pd.DataFrame) -> pd.DataFrame:
    """
    Ungerichtete, ungewichtete Kanten (u<v) aus (src,dst[,weight]).
    Entfernt Self-Loops & Duplikate.
    """
    colmap = {c.lower(): c for c in E.columns}
    src = next((colmap[k] for k in ("src","source","from","from_email","u","src_person_id") if k in colmap), None)
    dst = next((colmap[k] for k in ("dst","target","to","to_email","v","dst_person_id") if k in colmap), None)
    if src is None or dst is None:
        raise ValueError(f"src/dst Spalten nicht gefunden: {list(E.columns)}")

    if "weight" in colmap:
        w = colmap["weight"]
        E = E[[src, dst, w]].copy()
        E[w] = pd.to_numeric(E[w], errors="coerce").fillna(0.0)
        E = E[E[w] > 0]
    else:
        E = E[[src, dst]].copy()

    E = E.dropna(subset=[src, dst])
    E[src] = E[src].astype(str); E[dst] = E[dst].astype(str)
    E = E[E[src] != E[dst]]

    uv = pd.DataFrame({
        "u": np.where(E[src] < E[dst], E[src], E[dst]),
        "v": np.where(E[src] < E[dst], E[dst], E[src]),
    })
    return uv.drop_duplicates()

def _degree_and_index(uv: pd.DataFrame) -> Tuple[pd.Index, pd.Series, int]:
    nodes = pd.unique(uv[["u","v"]].values.ravel("K"))
    nodes = pd.Index(nodes, name="node")
    deg = pd.Series(0, index=nodes, dtype=np.int64)
    vc_u = uv["u"].value_counts()
    vc_v = uv["v"].value_counts()
    deg.loc[vc_u.index] += vc_u.astype(np.int64)
    deg.loc[vc_v.index] += vc_v.astype(np.int64)
    m_total = int(len(uv))
    return nodes, deg, m_total

def _cluster_members(H: pd.DataFrame, level_name: str) -> dict[int, list[str]]:
    if level_name not in H.columns:
        raise ValueError(f"{level_name} nicht in H: {list(H.columns)}")
    df = H[["node", level_name]].dropna().copy()
    df["node"] = df["node"].astype(str)
    df[level_name] = pd.to_numeric(df[level_name], errors="coerce").astype("Int64")
    groups = df.groupby(level_name)["node"].apply(list)
    return {int(cid): lst for cid, lst in groups.items()}

def _m_in_observed(uv: pd.DataFrame, members: set[str]) -> int:
    inS_u = uv["u"].isin(members)
    inS_v = uv["v"].isin(members)
    return int((inS_u & inS_v).sum())

def _expect_var_chung_lu_regularized(
    deg: pd.Series, m_total: int, members: list[str], eps: float = 1e-9,
    sample_pairs_cap: int = 500_000, rng_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Erwartung/Varianz im grad-korrigierten Konfigurationsmodell:
      p_ij = clip(k_i*k_j/(2m), eps, 1-eps)
      E[m_in] = sum_{i<j} p_ij
      Var ≈ sum_{i<j} p_ij (1 - p_ij)
    Für sehr große Cluster wird auf Paar-Sampling zurückgegriffen.
    """
    if m_total <= 0 or len(members) < 2:
        return 0.0, 1e-9, eps

    ks = deg.reindex(members).fillna(0).to_numpy(dtype=np.float64)
    n = ks.size
    pair_count = n * (n - 1) // 2

    if pair_count > 2_000_000:
        rng = np.random.default_rng(rng_seed)
        s = min(sample_pairs_cap, pair_count)
        # ziehe s zufällige Paar-Indizes (i<j) näherungsweise
        i = rng.integers(0, n, size=s)
        j = rng.integers(0, n, size=s)
        mask = i != j
        i, j = i[mask], j[mask]
        p = (ks[i] * ks[j]) / (2.0 * m_total)
        p = np.clip(p, eps, 1.0 - eps)
        Em = float(p.mean() * pair_count)
        Var = float((p * (1.0 - p)).mean() * pair_count)
        return Em, max(Var, 1e-9), eps

    # exakte Summe
    i_idx, j_idx = np.triu_indices(n, k=1)
    p = (ks[i_idx] * ks[j_idx]) / (2.0 * m_total)
    p = np.clip(p, eps, 1.0 - eps)
    Em = float(p.sum())
    Var = float((p * (1.0 - p)).sum())
    return Em, max(Var, 1e-9), eps

def _minsize_for(level_name: str, cfg: dict, stats_csv: Path | None) -> int:
    if stats_csv is not None and stats_csv.exists():
        ST = pd.read_csv(stats_csv).set_index("level")
        if level_name in ST.index and "min_size" in ST.columns:
            return int(ST.loc[level_name, "min_size"])
    fb = cfg.get("thresholds", {}).get("min_size_fallback", {"L1":5,"L2":5,"L3":5})
    i = int(level_name.split("_")[1])
    key = f"L{i}" if f"L{i}" in fb else "L3"
    return int(fb.get(key, 5))

# ---------- Hauptfunktion ----------
def run_significance(cfg: dict, use_prepared_edges: bool = True, eps: float = 1e-9) -> pd.DataFrame:
    """
    Rechnet robuste Z-Scores pro Cluster/Level und schreibt CSV nach outputs.signif_dir.
    Nutzt standardmäßig 'paths.edges_prepared' (empfohlen).
    Returns: Top-10-Summary über alle Level (DataFrame).
    """
    # Daten laden
    H = load_levels(cfg["paths"]["levels"])
    E_path = cfg["paths"]["edges_prepared"] if use_prepared_edges else cfg["paths"]["edges"]
    E = pd.read_csv(E_path)
    uv = _normalize_edges(E)
    nodes, deg, m_total = _degree_and_index(uv)

    level_cols = sorted([c for c in H.columns if str(c).startswith("level_")],
                        key=lambda x: int(str(x).split("_")[1]))
    out_dir = Path(cfg["outputs"]["signif_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = Path(cfg["outputs"]["cutoffs_dir"]) / "cluster_stats_summary.csv"

    summary_rows = []
    for lv in level_cols:
        ms = _minsize_for(lv, cfg, stats_csv)
        members_map = _cluster_members(H, lv)

        rows = []
        for cid, mem in members_map.items():
            n = len(mem)
            if n < ms:
                continue
            S = set(mem)
            m_in = _m_in_observed(uv, S)
            Em, Var, eps_used = _expect_var_chung_lu_regularized(deg, m_total, mem, eps=eps)

            z = (m_in - Em) / (Var**0.5 if Var > 0 else 1e-9)

            rows.append(dict(
                cluster_id=int(cid),
                n_nodes=int(n),
                pairs_possible=int(n*(n-1)//2),
                m_in_observed=int(m_in),
                exp_m_in=float(Em),
                var_m_in=float(Var),
                density_observed=float(m_in / max(1, n*(n-1)//2)),
                z_score=float(z),
                eps_used=float(eps_used),
                var_floor_used=1e-9 if Var<=1e-9 else 0.0,
            ))

        df = pd.DataFrame(rows).sort_values(["z_score","n_nodes"], ascending=[False, False])
        (out_dir / f"cluster_significance_{lv}.csv").write_text(df.to_csv(index=False))

        # Top-10 pro Level für schnelle Sichtprüfung
        if not df.empty:
            summary_rows.append(
                df.head(10)[["cluster_id","n_nodes","z_score","density_observed"]].assign(level=lv)
            )

    if summary_rows:
        SDF = pd.concat(summary_rows, ignore_index=True)
        (out_dir / "top10_per_level.csv").write_text(SDF.to_csv(index=False))
        return SDF
    else:
        # leere Summary zurückgeben
        return pd.DataFrame(columns=["cluster_id","n_nodes","z_score","density_observed","level"])