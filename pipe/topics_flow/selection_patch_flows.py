"""
selection_patch_flows.py
------------------------
Sorgt dafür, dass alle Cluster-Endpunkte aus den Topic-Flows in der Hierarchie (selection_nested)
enthalten sind. Nutzt numerische IDs (z. B. level_4) aus infomap_levels.csv.
"""

import json
import pandas as pd
import itertools as it
from pathlib import Path
from copy import deepcopy
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# ---------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------

def patch_selection_with_flows(cfg_path: str | Path, org_dir: str | Path,
                               selection_nested: dict,
                               max_k: int = 5,
                               flow_level: int = 4) -> tuple[dict, list]:
    """
    Ergänzt selection_nested um alle Knoten, die in topic_flows_filtered.csv auf L{flow_level} vorkommen.
    Gibt (selection_patched, extra_edges) zurück.
    """

    from yaml import safe_load
    cfg = safe_load(Path(cfg_path).read_text())
    LEVELS_CSV = Path(cfg["paths"]["levels"])
    FLOWS_CSV  = Path(org_dir) / "topic_flows_filtered.csv"
    assert LEVELS_CSV.exists(), f"Fehlt: {LEVELS_CSV}"
    assert FLOWS_CSV.exists(), f"Fehlt: {FLOWS_CSV}"

    # --- Hilfsfunktionen ---
    def ensure_all_levels(sel: dict, max_k=5):
        have = {k for k in sel.keys() if str(k).upper().startswith("L")}
        need = {f"L{i}" for i in range(1, max_k+1)}
        for k in sorted(need - have, key=lambda x: int(x[1:])):
            sel[k] = []
        return {f"L{i}": sel.get(f"L{i}", []) for i in range(1, max_k+1)}

    def mp_prefix(mp: str, k: int) -> str:
        parts = str(mp).split(":")
        return ":".join(parts[:k])

    # --- Daten laden ---
    H = pd.read_csv(LEVELS_CSV)
    flows = pd.read_csv(FLOWS_CSV)
    selection_nested = ensure_all_levels(selection_nested, max_k=max_k)

    # --- Mapping module_path → level_k ID ---
    col_level = f"level_{flow_level}"
    mp2id = {}
    for _, r in H[["module_path", col_level]].dropna().iterrows():
        mpk = mp_prefix(r["module_path"], flow_level)
        mp2id[mpk] = int(r[col_level])

    # --- Farbskala je Topic ---
    topic_ids = sorted(flows["thread_topic_id"].dropna().astype(int).unique().tolist())
    cmap = cm.get_cmap("tab20", max(10, len(topic_ids)))
    topic_color = {tid: mcolors.to_hex(cmap(i % cmap.N)) for i, tid in enumerate(topic_ids)}

    # --- Flüsse auf Lk mappen ---
    rows = []
    for _, r in flows.iterrows():
        s = mp2id.get(mp_prefix(str(r["sender_cluster"]), flow_level))
        d = mp2id.get(mp_prefix(str(r["recipient_cluster"]), flow_level))
        if s is None or d is None or s == d:
            continue
        rows.append((f"L{flow_level}", s, d, int(r["thread_topic_id"]), float(r["weight"])))

    extra_edges = []
    for (lv, s, d, tid), grp in it.groupby(sorted(rows), key=lambda x: (x[0], x[1], x[2], x[3])):
        w = sum(g[4] for g in grp)
        extra_edges.append(dict(
            level=lv, src=s, dst=d,
            width=max(1.5, w / 15000.0),
            color=topic_color.get(tid, "#cc6600"),
            label=f"topic {tid} • {int(w)}",
            arrows="to", smooth=True, dashes=True,
        ))

    print(f"[flows→edges] mapped: {len(extra_edges)} (L{flow_level})")

    # --- Elternbeziehungen ---
    parents = {}
    for _, r in H.dropna(subset=["level_1"]).iterrows():
        for i in range(1, max_k):
            a, b = f"level_{i}", f"level_{i+1}"
            if a in r and b in r and pd.notna(r[b]):
                parents.setdefault((f"L{i+1}", int(r[b])), (f"L{i}", int(r[a])))

    # --- Indexe & Ergänzungsfunktionen ---
    def index_children(sel):
        idx = {f"L{i}": {} for i in range(1, max_k+1)}
        def add(level, node):
            idx[level][node["cid"]] = node
            nxt = f"L{int(level[1:])+1}"
            for ch in node.get("children", []) or []:
                add(nxt, ch)
        for n in sel.get("L1", []):
            add("L1", n)
        return idx

    def ensure_chain(sel, chain):
        idx = index_children(sel)
        # Root
        L1, id1 = chain[0]
        if id1 not in idx[L1]:
            sel[L1].append({"cid": id1, "keep": True, "children": []})
            idx = index_children(sel)
        # nach unten
        for i in range(len(chain) - 1):
            Lp, pid = chain[i]
            Lc, cid = chain[i + 1]
            pnode = idx[Lp][pid]
            if cid not in [c["cid"] for c in pnode.get("children", [])]:
                pnode.setdefault("children", []).append({"cid": cid, "keep": True, "children": []})
                idx = index_children(sel)

    def chain_to_L1(Lk, cid):
        chain = [(Lk, cid)]
        cur = (Lk, cid)
        while cur in parents:
            cur = parents[cur]
            chain.insert(0, cur)
        return chain

    # --- Patch anwenden ---
    selection_patched = deepcopy(selection_nested)
    needed_Lk = {e["src"] for e in extra_edges if e["level"] == f"L{flow_level}"} | \
                {e["dst"] for e in extra_edges if e["level"] == f"L{flow_level}"}

    for cid in needed_Lk:
        chain = chain_to_L1(f"L{flow_level}", cid)
        ensure_chain(selection_patched, chain)

    print(f"[selection] patched with {len(needed_Lk)} L{flow_level}-nodes (flows-endpoints)")
    return selection_patched, extra_edges