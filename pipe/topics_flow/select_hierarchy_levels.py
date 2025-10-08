"""
pipe.select_hierarchy_levels
============================
Analysiert die hierarchischen Infomap-Level-Daten und wählt dynamisch
repräsentative Cluster auf jeder Ebene aus (basierend auf Größenstatistik
und Coverage-Ziel). Erzeugt eine reduzierte Auswahlstruktur zur späteren
Visualisierung oder zum Export.

Inputs:
  - data_derived/infomap_levels.csv
  - (optional) reports/orgchart/cutoffs/cluster_stats_summary.csv
  - (optional) reports/orgchart/cutoffs/parent_child_*.csv

Outputs:
  - reports/orgchart/selection_dynamic.json
  - reports/orgchart/selection_meta.json
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np


def select_hierarchy_levels(env: dict) -> None:
    org_dir = Path(env["outputs"]["org_dir"])
    cut_dir = Path(env["outputs"].get("cutoffs_dir", org_dir / "cutoffs"))
    cut_dir.mkdir(parents=True, exist_ok=True)

    levels_csv = Path(env["inputs"].get("levels_csv",
                      org_dir.parent / "data_derived" / "infomap_levels.csv"))
    assert levels_csv.exists(), f"infomap_levels.csv fehlt: {levels_csv}"

    H = pd.read_csv(levels_csv)
    level_cols = [c for c in H.columns if str(c).startswith("level_")]
    assert level_cols, f"Keine level_* Spalten in {levels_csv.name}"
    level_cols = sorted(level_cols, key=lambda s: int(str(s).split("_")[1]))

    # === Hilfsfunktionen ===
    def _coverage_topk(counts: pd.Series, target: float = 0.8) -> list:
        v = counts.sort_values(ascending=False)
        if v.sum() == 0:
            return []
        c = (v.cumsum() / v.sum()).values
        k = int(np.searchsorted(c, target) + 1)
        return list(v.index[:k])

    def _pick_min_size(sizes: pd.Series) -> int:
        if len(sizes) == 0:
            return 5
        p95 = int(np.percentile(sizes.values, 95))
        return max(5, min(20, p95))

    def _build_parent_child(df: pd.DataFrame, parent_lv: str, child_lv: str, cap_children: int = 30) -> dict:
        grp = df.groupby([parent_lv, child_lv]).size().reset_index(name="n").dropna()
        mp = {}
        for p, sub in grp.groupby(parent_lv):
            kids = sub.sort_values("n", ascending=False)[child_lv].astype(object).tolist()[:cap_children]
            mp[p] = kids
        return mp

    # === Analyse ===
    selection = {"levels": level_cols, "keep": {}, "children": {}, "meta": {}}

    for i, LV in enumerate(level_cols):
        sizes = H[LV].dropna().astype(object).value_counts()
        min_size = _pick_min_size(sizes)
        sizes_f = sizes[sizes >= min_size]
        keep_ids = _coverage_topk(sizes_f, target=0.8)
        selection["keep"][LV] = [int(x) if str(x).isdigit() else x for x in keep_ids]
        selection["meta"][LV] = {
            "min_size": int(min_size),
            "coverage": 0.8,
            "kept": len(keep_ids),
            "clusters_total": len(sizes)
        }

        if i < len(level_cols) - 1:
            P, C = LV, level_cols[i + 1]
            selection["children"][C] = _build_parent_child(H[H[P].isin(keep_ids)], P, C)

    # === Persistenz ===
    sel_json = org_dir / "selection_dynamic.json"
    meta_json = org_dir / "selection_meta.json"
    sel_json.write_text(json.dumps(selection, indent=2))
    meta_json.write_text(json.dumps(selection["meta"], indent=2))
    print(f"[write] {sel_json}")
    print(f"[write] {meta_json}")

    # === Checks ===
    for LV in level_cols:
        print(f"{LV}: kept={len(selection['keep'][LV])}/{selection['meta'][LV]['clusters_total']} "
              f"| min_size={selection['meta'][LV]['min_size']}")
    print("[ok] selection ist render-bereit")


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    select_hierarchy_levels(env)