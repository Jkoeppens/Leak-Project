"""
pipe.selection_to_nested
========================
Konvertiert die flache Auswahlstruktur aus selection_dynamic.json
(in der Levels separat gelistet sind) in eine verschachtelte Baumstruktur.

Input:
  - reports/orgchart/selection_dynamic.json

Output:
  - reports/orgchart/selection_nested.json
"""

import json
from pathlib import Path


def selection_to_nested(env: dict) -> dict:
    org_dir = Path(env["outputs"]["org_dir"])
    sel_json = org_dir / "selection_dynamic.json"
    assert sel_json.exists(), f"selection_dynamic.json fehlt: {sel_json}"

    sel_raw = json.loads(sel_json.read_text())

    # 1) Level-Listen normalisieren
    level_cols = sel_raw.get("levels", [])
    Lkeys = [f"L{str(c).split('_')[1]}" if "_" in str(c) else str(c) for c in level_cols]

    # 2) Keep-Maps je Level
    keep_map = {
        Lkeys[i]: set(sel_raw["keep"].get(level_cols[i], []))
        for i in range(len(level_cols))
    }

    # 3) Parent->Child-Mapping (remappt auf (L1,L2)...)
    children_map = {}
    for i in range(len(level_cols) - 1):
        parent_L = Lkeys[i]
        child_L = Lkeys[i + 1]
        mp_raw = sel_raw.get("children", {}).get(level_cols[i + 1], {})
        mp = {}
        for k, v in mp_raw.items():
            try:
                pk = int(k)
            except Exception:
                pk = k
            kids = []
            for x in (v or []):
                try:
                    kids.append(int(x))
                except Exception:
                    kids.append(x)
            mp[pk] = kids
        children_map[(parent_L, child_L)] = mp

    # 4) Rekursive Funktion
    def build_node(level_idx: int, cid):
        L = Lkeys[level_idx]
        node = {"cid": cid, "keep": True, "children": []}
        if level_idx >= len(Lkeys) - 1:
            return node
        parent_L = Lkeys[level_idx]
        child_L = Lkeys[level_idx + 1]
        mp = children_map.get((parent_L, child_L), {})
        kids = mp.get(cid, [])
        keep_next = keep_map.get(child_L, set())
        node["children"] = [
            build_node(level_idx + 1, k) for k in kids if k in keep_next
        ]
        return node

    # 5) Baum bauen
    top_L = Lkeys[0]
    top_ids = sorted(list(keep_map.get(top_L, set())))
    selection_nested = {top_L: [build_node(0, cid) for cid in top_ids]}

    # 6) Diagnostik & Persistenz
    def _count_nodes(lst):
        return sum(1 + _count_nodes(d.get("children", [])) for d in lst)

    n_total = _count_nodes(selection_nested[top_L])
    out_path = org_dir / "selection_nested.json"
    out_path.write_text(json.dumps(selection_nested, indent=2))
    print(f"[write] {out_path} | Ebenen: {Lkeys} | Root={len(top_ids)} | total≈{n_total}")
    return selection_nested


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    selection_to_nested(env)