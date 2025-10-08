# pipe/io.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict  # oder lass Dict weg und nutze 'dict'
import pandas as pd
import numpy as np

# ---------- intern ----------
def _pick(cols, cands):
    for c in cands:
        if c in cols: return c
    lower = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in lower: return lower[c.lower()]
    return None

def _is_internal_series(series: pd.Series, domains: List[str]) -> pd.Series:
    patt = tuple(domains or [])
    if not patt:
        return pd.Series(True, index=series.index)
    return series.astype(str).str.endswith(patt)

# ---------- Loader ----------
def load_edges_prepared(cfg: dict) -> pd.DataFrame:
    p = Path(cfg["paths"]["edges_prepared"])
    if not p.exists():
        raise FileNotFoundError(f"edges_prepared nicht gefunden: {p}")
    df = pd.read_csv(p)
    required = {"src", "dst", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"edges_prepared fehlt Spalten: {missing}")
    return df

def load_levels(levels_csv: str) -> pd.DataFrame:
    H = pd.read_csv(levels_csv)
    level_cols = [c for c in H.columns if str(c).startswith("level_")]
    for c in level_cols:
        H[c] = pd.to_numeric(H[c], errors="coerce").astype("Int64")
    if "node" not in H.columns:
        node_col = _pick(H.columns, ["node","person_id","email"])
        if node_col is None:
            raise ValueError("levels.csv braucht 'node' (oder person_id/email).")
        H = H.rename(columns={node_col:"node"})
    H["node"] = H["node"].astype(str)
    return H

def load_edges(edges_csv: str) -> pd.DataFrame:
    E = pd.read_csv(edges_csv)
    src = _pick(E.columns, ["src","src_person_id","source","from_email","from","u"])
    dst = _pick(E.columns, ["dst","dst_person_id","target","to_email","to","v"])
    w   = _pick(E.columns, ["weight","w","count","n","value"])
    if src is None or dst is None:
        raise ValueError(f"src/dst Spalten nicht gefunden in {list(E.columns)}")
    if w is None:
        E["weight"] = 1.0; w = "weight"
    E = E[[src,dst,w]].rename(columns={src:"src", dst:"dst", w:"weight"})
    E["src"] = E["src"].astype(str)
    E["dst"] = E["dst"].astype(str)
    E["weight"] = pd.to_numeric(E["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    E = E.dropna(subset=["src","dst"])
    E = E[E["src"] != E["dst"]]
    E = (E.groupby(["src","dst"], as_index=False)["weight"]
           .sum()
           .sort_values("weight", ascending=False))
    return E

# ---------- Extern-/Filter-Logik ----------
def mark_outside(E: pd.DataFrame, domains: List[str], outside_label: str = "Outside world") -> pd.DataFrame:
    src_int = _is_internal_series(E["src"], domains)
    dst_int = _is_internal_series(E["dst"], domains)
    F = E.copy()
    F.loc[~src_int, "src"] = outside_label
    F.loc[~dst_int, "dst"] = outside_label
    return F.groupby(["src","dst"], as_index=False)["weight"].sum()

def filter_internal_mode(E: pd.DataFrame, mode: str, domains: List[str], outside_label: str = "Outside world") -> pd.DataFrame:
    m = (mode or "outside_supernode").lower()
    if m == "include":
        return E.copy()
    if m == "outside_supernode":
        return mark_outside(E, domains, outside_label)
    if m == "exclude":
        src_int = _is_internal_series(E["src"], domains)
        dst_int = _is_internal_series(E["dst"], domains)
        return E[src_int & dst_int].copy()
    raise ValueError(f"Unbekannter mode={mode}. Erlaubt: include | outside_supernode | exclude")

def apply_edge_filters(E: pd.DataFrame, min_weight: int = 1, require_reciprocal: bool = False) -> pd.DataFrame:
    F = E.copy()
    if min_weight and min_weight > 1:
        F = F[F["weight"] >= float(min_weight)]
    if require_reciprocal:
        pairs = set(map(tuple, F[["src","dst"]].values))
        recip = pd.DataFrame([(u,v) for (u,v) in pairs if (v,u) in pairs], columns=["src","dst"])
        F = F.merge(recip, on=["src","dst"], how="inner") if len(recip) else F.iloc[0:0]
    return F

def ego_subgraph(E: pd.DataFrame, seeds: List[str], hops: int = 0) -> pd.DataFrame:
    if not seeds or hops <= 0:
        return E.copy()
    seeds = set(map(str, seeds))
    nbr = set(seeds); cur = set(seeds)
    for _ in range(hops):
        sub = E[E["src"].isin(cur) | E["dst"].isin(cur)]
        nxt = set(sub["src"]).union(set(sub["dst"]))
        cur = nxt - nbr
        nbr |= nxt
    nbr = set(map(str, nbr))
    return E[E["src"].isin(nbr) & E["dst"].isin(nbr)].copy()

# ---------- Reports & Plots ----------
def basic_report(E: pd.DataFrame, H: pd.DataFrame | None = None) -> dict:
    nodes = pd.unique(E[["src","dst"]].values.ravel("K"))
    n, m = len(nodes), len(E)
    deg = pd.Series(0, index=pd.Index(nodes, name="node"), dtype=int)
    for col in ["src","dst"]:
        c = E[col].value_counts()
        deg.loc[c.index] += c.astype(int)
    deg = deg.sort_values(ascending=False)
    rep = {
        "nodes": int(n),
        "edges": int(m),
        "weight_sum": float(E["weight"].sum()),
        "weight_min": float(E["weight"].min() if m else 0.0),
        "weight_max": float(E["weight"].max() if m else 0.0),
        "degree_max": int(deg.iloc[0]) if len(deg) else 0,
        "degree_p95": float(deg.quantile(0.95)) if len(deg) else 0.0,
        "top_degree_nodes": deg.head(10).to_dict(),
    }
    if H is not None:
        for lv in [c for c in H.columns if str(c).startswith("level_")]:
            rep[f"{lv}_clusters"] = int(H[lv].nunique(dropna=True))
    return rep

def save_report(report: dict, outdir: str):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    (out/"data_report.json").write_text(pd.Series(report).to_json(indent=2))
    td = pd.DataFrame(list(report.get("top_degree_nodes", {}).items()), columns=["node","degree"])
    if len(td): td.to_csv(out/"top_degree_nodes.csv", index=False)

def hist_weight(E: pd.DataFrame, out_png: str):
    import matplotlib.pyplot as plt
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    w = E["weight"].astype(float)
    plt.figure(figsize=(6,4))
    plt.hist(w, bins=50, log=True)
    plt.xlabel("Gewicht"); plt.ylabel("Anzahl (log)")
    plt.title("Weight-Histogramm (prepared)")
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()

def hist_degree(E: pd.DataFrame, out_png: str):
    import matplotlib.pyplot as plt
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    nodes = pd.unique(E[["src","dst"]].values.ravel("K"))
    deg = pd.Series(0, index=pd.Index(nodes, name="node"), dtype=int)
    for col in ["src","dst"]:
        c = E[col].value_counts()
        deg.loc[c.index] += c.astype(int)
    plt.figure(figsize=(6,4))
    plt.hist(deg.values, bins=60, log=True)
    plt.xlabel("Degree"); plt.ylabel("Anzahl Knoten (log)")
    plt.title("Degree-Histogramm (prepared, ungerichtet)")
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()
