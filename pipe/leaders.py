# pipe/leaders.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any
import json
import numpy as np
import pandas as pd
import networkx as nx

OUTSIDE = "Outside world"

def _undirected_weighted(E: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for r in E.itertuples(index=False):
        u, v, w = r.src, r.dst, float(r.weight)
        if w <= 0:
            continue
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G

def _is_internal(name: str, domains: tuple[str, ...]) -> bool:
    if not name or name == OUTSIDE:
        return False
    return name.endswith(domains) if domains else True

def _filter_candidates(nodes, mode: str, domains: tuple[str, ...], allow_external: bool) -> list[str]:
    nodes = set(map(str, nodes))
    mode = (mode or "outside_supernode").lower()
    if mode == "include":
        return [n for n in nodes if allow_external or _is_internal(n, domains)]
    # outside_supernode & exclude -> nur interne
    return [n for n in nodes if _is_internal(n, domains)]

def _leader_for_nodes(
    E: pd.DataFrame,
    nodes,
    cap: int = 800,
    seed: int = 42,
    mode: str = "outside_supernode",
    domains: tuple[str, ...] = tuple(),
    allow_external_leaders: bool = False,
) -> tuple[str | None, float]:
    cand = _filter_candidates(nodes, mode, domains, allow_external_leaders)
    if not cand:
        return None, 0.0
    sub = E[E["src"].isin(cand) & E["dst"].isin(cand)]
    if sub.empty:
        return None, 0.0
    Gm = _undirected_weighted(sub)
    kk = min(cap, Gm.number_of_nodes()) if cap else Gm.number_of_nodes()
    bc = nx.betweenness_centrality(Gm, k=kk, normalized=True, seed=seed)
    if not bc:
        return None, 0.0
    u, sc = max(bc.items(), key=lambda x: x[1])
    return u, float(sc)

def _members(H: pd.DataFrame, level: str, cid: int) -> list[str]:
    return H.loc[H[level] == cid, "node"].astype(str).tolist()

def compute_leaders(
    H: pd.DataFrame,
    E: pd.DataFrame,
    selection: dict,
    *,
    pct_exec: float = 99.9,
    bt_cap: int = 2000,
    seed: int = 42,
    internal_mode: str = "outside_supernode",
    internal_domains: tuple[str, ...] = tuple(),
    allow_external_leaders: bool = False,
) -> dict[str, Any]:
    """
    Ermittelt:
      - globale Execs (Top-Betweenness, nur intern)
      - je Cluster (pro Level) eine(n) Leader via Betweenness auf dem induzierten Subgraph

    Args:
      H: DataFrame mit 'node' und level_* Spalten
      E: Kanten (src, dst, weight), bereits nach deiner Policy vorbereitet
      selection: dict mit keys ['levels','keep','children'] (aus dynamic selection)
    """
    levels = selection["levels"]

    # globale Execs (nur intern)
    Gu = _undirected_weighted(E)
    kk = min(bt_cap, Gu.number_of_nodes()) if bt_cap else Gu.number_of_nodes()
    bc = nx.betweenness_centrality(Gu, k=kk, normalized=True, seed=seed)
    thr = np.percentile(list(bc.values()), pct_exec) if bc else 1.0
    exec_nodes = [u for u, s in bc.items() if s >= thr and _is_internal(u, internal_domains)]

    leaders: dict[str, Any] = {"execs": exec_nodes, "levels": {lv: {} for lv in levels}}
    L1 = levels[0]

    # Ebene 1
    for l1 in selection["keep"].get(L1, []):
        nodes_L1 = _members(H, L1, int(l1))
        lead, score = _leader_for_nodes(
            E, nodes_L1, cap=800, seed=seed,
            mode=internal_mode, domains=internal_domains,
            allow_external_leaders=allow_external_leaders
        )
        leaders["levels"][L1][int(l1)] = {"leader": lead, "score": score}

    # tiefere Ebenen
    for i in range(1, len(levels)):
        parent_lv = levels[i - 1]
        child_lv = levels[i]
        for parent_id, child_ids in selection["children"].get(child_lv, {}).items():
            for cid in child_ids:
                nodes = _members(H, child_lv, int(cid))
                lead, score = _leader_for_nodes(
                    E, nodes, cap=800, seed=seed,
                    mode=internal_mode, domains=internal_domains,
                    allow_external_leaders=allow_external_leaders
                )
                leaders["levels"][child_lv][int(cid)] = {"leader": lead, "score": score}

    return leaders

def flatten_leaders(leaders: dict) -> pd.DataFrame:
    flat = []
    for lv, mp in leaders["levels"].items():
        for cid, info in mp.items():
            flat.append({"level": lv, "cluster_id": int(cid), "leader": info.get("leader"), "score": info.get("score")})
    return pd.DataFrame(flat).sort_values(["level", "cluster_id"])

def save_leaders(leaders: dict, cfg: dict) -> tuple[str, str]:
    out_dir = Path(cfg["outputs"]["org_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    j1 = out_dir / "leaders.json"
    j1.write_text(json.dumps(leaders, indent=2, ensure_ascii=False))
    flat = flatten_leaders(leaders)
    csv = out_dir / "leaders_flat.csv"
    flat.to_csv(csv, index=False)
    return str(j1), str(csv)

def find_external_leaders(leaders: dict, internal_domains: tuple[str, ...]) -> list[tuple[str, int, str]]:
    def _is_internal_email(name: str) -> bool:
        return bool(name) and (name != OUTSIDE) and name.endswith(internal_domains)
    bad = []
    for lv, mp in leaders["levels"].items():
        for cid, info in mp.items():
            if info.get("leader") and not _is_internal_email(info["leader"]):
                bad.append((lv, int(cid), info["leader"]))
    return bad
