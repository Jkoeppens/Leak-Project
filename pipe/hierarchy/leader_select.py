# pipe/leader_select.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple
import json
import numpy as np
import pandas as pd
import networkx as nx

from .io import load_levels  # robustes Level-Loading

# ---------- Helpers ----------
def _pick_col(cols: Iterable[str], *cands: Iterable[str]) -> str | None:
    lc = {c.lower(): c for c in cols}
    for group in cands:
        for name in group:
            if name.lower() in lc:
                return lc[name.lower()]
    return None

def _undirected_simple_edges(E: pd.DataFrame, src: str, dst: str) -> pd.DataFrame:
    X = E[[src, dst]].dropna().astype(str)
    X = X[X[src] != X[dst]]
    uv = pd.DataFrame({
        "u": np.where(X[src] < X[dst], X[src], X[dst]),
        "v": np.where(X[src] < X[dst], X[dst], X[src]),
    }).drop_duplicates()
    return uv

def _visible_clusters(selection: dict) -> dict[str, set[int]]:
    levels = selection.get("levels", [])
    vis = {lv: set() for lv in levels}
    if not levels:
        return vis
    L1 = levels[0]
    vis[L1] = set(map(int, selection.get("keep", {}).get(L1, [])))
    for i in range(1, len(levels)):
        p_lv, c_lv = levels[i-1], levels[i]
        cmap = selection.get("children", {}).get(c_lv, {})
        keep = set()
        for p, kids in cmap.items():
            if int(p) in vis[p_lv]:
                keep.update(map(int, kids))
        vis[c_lv] = keep
    return vis

def _pick_leader_degree(G: nx.Graph, members: list[str]) -> tuple[str | None, int | None, int | None]:
    nodes = [n for n in members if n in G]
    if not nodes:
        return None, None, None
    S = G.subgraph(nodes)
    deg_in = dict(S.degree())
    deg_gl = dict(G.degree(nodes))
    leader = max(nodes, key=lambda n: (deg_in.get(n, 0),
                                       deg_gl.get(n, 0),
                                       -len(n),
                                       n))
    return leader, int(deg_in.get(leader, 0)), int(deg_gl.get(leader, 0))

# ---------- Hauptlogik ----------
def build_leaders_by_degree(cfg: dict, *, use_prepared_edges: bool = True) -> dict[str, Any]:
    """
    Wählt Leader je sichtbarerm Cluster via 'max internes Degree im Cluster'.
    Zusätzlich: 'execs' = Top-n_execs nach globalem Degree.
    Schreibt leaders.json & leaders_flat.csv in outputs.org_dir.
    Returns: leaders-dict.
    """
    # Pfade
    org_dir = Path(cfg["outputs"]["org_dir"]); org_dir.mkdir(parents=True, exist_ok=True)
    sel_path = org_dir / "selection_dynamic.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"selection_dynamic.json fehlt: {sel_path}")

    # Daten
    H = load_levels(cfg["paths"]["levels"])
    E_path = cfg["paths"]["edges_prepared"] if use_prepared_edges else cfg["paths"]["edges"]
    E = pd.read_csv(E_path)

    # Spalten finden & Graph bauen
    src = _pick_col(E.columns,
                    ("src_person_id","src","source","from","from_email","u","sender","author"))
    dst = _pick_col(E.columns,
                    ("dst_person_id","dst","target","to","to_email","v","recipient","receiver"))
    if src is None or dst is None:
        raise RuntimeError(f"src/dst nicht gefunden in Edges-Spalten: {list(E.columns)}")

    uv = _undirected_simple_edges(E, src, dst)
    G = nx.Graph(); G.add_edges_from(uv.itertuples(index=False, name=None))

    # Selection & sichtbare Cluster
    selection = json.loads(sel_path.read_text())
    levels = selection.get("levels", [])
    if not levels:
        raise RuntimeError("selection_dynamic.json enthält keine 'levels'.")
    visible = _visible_clusters(selection)

    # Execs
    n_execs = int(cfg.get("thresholds", {}).get("n_execs", 8))
    execs = [n for n,_deg in sorted(G.degree, key=lambda x: x[1], reverse=True)[:n_execs]]

    # Leader je Level/Cluster
    leaders_out: dict[str, Any] = {"execs": execs, "levels": {}}
    for lv in levels:
        leaders_out["levels"].setdefault(lv, {})
        cids = sorted(map(int, visible.get(lv, set())))
        for cid in cids:
            members = H.loc[H[lv] == cid, "node"].astype(str).tolist()
            who, din, dgl = _pick_leader_degree(G, members)
            if who is None:
                # kein Eintrag schreiben, aber weitermachen
                continue
            leaders_out["levels"][lv][cid] = {"leader": who, "deg_in": din, "deg_global": dgl}

    # Speichern
    jpath = org_dir / "leaders.json"
    jpath.write_text(json.dumps(leaders_out, indent=2, ensure_ascii=False))

    flat = []
    for lv, mp in leaders_out["levels"].items():
        for cid, info in mp.items():
            flat.append({"level": lv, "cluster_id": int(cid),
                         "leader": info.get("leader"),
                         "deg_in": info.get("deg_in"),
                         "deg_global": info.get("deg_global")})
    pd.DataFrame(flat).sort_values(["level","cluster_id"]).to_csv(org_dir / "leaders_flat.csv", index=False)

    return leaders_out
