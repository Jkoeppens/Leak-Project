# pipe/orgchart_html.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from pyvis.network import Network


def _esc(s: str) -> str:
    if s is None:
        return "-"
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _name_map(persons: pd.DataFrame | None) -> dict[str, str]:
    """Erzeuge Mapping person_id/email -> Name (fallback: id selbst)."""
    if persons is None or persons.empty:
        return {}
    pid = None
    for c in persons.columns:
        if c in ("person_id", "node", "email"):
            pid = c
            break
    if not pid:
        return {}
    nm = "name" if "name" in persons.columns else pid
    return dict(zip(persons[pid].astype(str), persons[nm].astype(str)))


def _sizes_per_level(H: pd.DataFrame, levels: list[str]) -> dict[str, dict[int, int]]:
    """Clustergrößen je Level."""
    return {lv: H[lv].value_counts().astype(int).to_dict() for lv in levels}


def _colors_for(levels: list[str]) -> dict[str, str]:
    base = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]
    colors = {"EXEC": "#ff7f0e"}
    for i, lv in enumerate(levels, start=1):
        colors[lv] = base[(i - 1) % len(base)]
    return colors


def _prune_selection(selection: dict, max_levels: list[str]) -> dict:
    """Beschneidet selection auf die ersten max_levels Ebenen und filtert Children entlang sichtbarer Parents."""
    import json
    sel = json.loads(json.dumps(selection))  # deep copy
    sel["levels"] = list(max_levels)
    if not max_levels:
        return sel

    sel["keep"] = {lv: sel.get("keep", {}).get(lv, []) for lv in max_levels}
    children = sel.get("children", {}) or {}
    new_children, visible = {}, {lv: set() for lv in max_levels}

    # L1 sichtbar = keep[L1]
    visible[max_levels[0]] = set(map(int, sel["keep"].get(max_levels[0], [])))

    # Tiefer: nur Kinder von sichtbaren Eltern
    for d in range(1, len(max_levels)):
        parent_lv, child_lv = max_levels[d - 1], max_levels[d]
        cmap = children.get(child_lv, {}) or {}
        filtered = {}
        for p, kids in cmap.items():
            if int(p) in visible[parent_lv]:
                filtered[str(p)] = kids
        new_children[child_lv] = filtered
        vis_children = set()
        for kids in filtered.values():
            vis_children.update(map(int, kids))
        visible[child_lv] = vis_children

    sel["children"] = new_children
    return sel


def build_org_html(
    selection: dict,
    leaders: dict,
    H: pd.DataFrame,
    persons: pd.DataFrame | None = None,
    out_html: str = "organigram_interaktiv.html",
    max_depth: int = 3,
    z_by_level: dict | None = None,
    topics_by_level: dict | None = None,
    physics: bool = False,
    label_template: str = "{level}:{cid} • {leader} • n={n} • z={z:.1f} • k_in={deg_in} • k={deg_global} • {topics}",
    label_size: int = 18,
) -> str:
    """
    Erzeugt ein interaktives Orgchart (PyVis).
    - Topics erscheinen IMMER im Tooltip (falls übergeben) und optional im Label via {topics}.
    - z_by_level: dict[level][cluster_id] -> float
    - topics_by_level: dict[level][cluster_id] -> string (vorformatierter Mix)
    """

    # H normalisieren
    if "node" not in H.columns:
        for cand in ("person_id", "email", "node"):
            if cand in H.columns:
                H = H.rename(columns={cand: "node"})
                break
    H["node"] = H["node"].astype(str)

    levels_all = list(selection.get("levels", []))
    if not levels_all:
        raise ValueError("selection['levels'] fehlt/leer.")
    levels = levels_all[:max_depth]
    selection = _prune_selection(selection, levels)

    nmap = _name_map(persons)
    sizes = _sizes_per_level(H, levels)
    zmap = z_by_level or {}
    tmap = topics_by_level or {}
    COLORS = _colors_for(levels)

    net = Network(width="100%", height="920px", bgcolor="#ffffff", font_color="#111", directed=True, notebook=False)
    net.set_options(
        f"""
    {{
      "interaction": {{"hover": true, "navigationButtons": true}},
      "physics": {{ "enabled": {"true" if physics else "false"},
                    "stabilization": {{"iterations": 300}} }},
      "layout": {{ "hierarchical": {{ "enabled": true,
                                      "direction": "UD",
                                      "sortMethod": "directed",
                                      "levelSeparation": 180,
                                      "nodeSpacing": 120 }} }},
      "nodes": {{
        "shape": "dot",
        "scaling": {{ "min": 10, "max": 90 }},
        "font": {{
          "size": {label_size},
          "face": "arial",
          "color": "#111111",
          "strokeWidth": 5,
          "strokeColor": "#ffffff"
        }},
        "borderWidth": 1
      }},
      "edges": {{
        "arrows": {{"to": {{"enabled": false}}}},
        "color": {{"opacity": 0.28}}
      }}
    }}
    """
    )

    # Execs
    execs = list(leaders.get("execs", []))
    exec_ids = []
    for i, who in enumerate(execs, start=1):
        nid = f"EXEC:{i}"
        lbl = f"EXEC {i}"
        title = f"<b>Executive</b><br/>{_esc(nmap.get(who, who))}"
        net.add_node(nid, label=lbl, level=0, color=COLORS["EXEC"], value=70, title=title)
        exec_ids.append(nid)

    # Ebene 1
    L1 = levels[0]
    keep_L1 = selection.get("keep", {}).get(L1, [])
    for idx, l1 in enumerate(keep_L1):
        l1 = int(l1)
        n1 = f"{L1}:{l1}"
        sz = sizes.get(L1, {}).get(l1, 0)
        rec = (leaders.get("levels", {}) or {}).get(L1, {}).get(l1, {}) or {}
        lead_id = rec.get("leader")
        din = rec.get("deg_in", np.nan)
        dgl = rec.get("deg_global", np.nan)
        lead_nm = nmap.get(lead_id, lead_id)
        topics_txt = (tmap.get(L1, {}) or {}).get(l1, "")
        z_for_label = float("nan")

        # Label + Tooltip
        label = label_template.format(
            level=L1, cid=l1, leader=(lead_nm or "–"), z=z_for_label, n=sz, deg_in=din, deg_global=dgl, topics=topics_txt
        )
        tip = (
            f"<b>{L1} {l1}</b><br/>Persons: {sz}<br/>Leader: {_esc(lead_nm) if lead_nm else '–'}"
            f"<br/>k_in={'' if np.isnan(din) else int(din)} · k={'' if np.isnan(dgl) else int(dgl)}"
            f"<br/>Topics: {_esc(topics_txt) if topics_txt else '–'}"
        )

        val = min(90, 15 + int(np.sqrt(max(sz, 1))))
        net.add_node(n1, label=label, level=1, color=COLORS[L1], value=val, title=tip)
        if exec_ids:
            net.add_edge(exec_ids[idx % len(exec_ids)], n1)

    # tiefere Ebenen
    for depth in range(1, len(levels)):
        parent_lv = levels[depth - 1]
        child_lv = levels[depth]
        cmap = selection.get("children", {}).get(child_lv, {}) or {}
        for parent_id, child_ids in cmap.items():
            pnode = f"{parent_lv}:{int(parent_id)}"
            for cid in child_ids:
                cid = int(cid)
                cnode = f"{child_lv}:{cid}"
                sz = sizes.get(child_lv, {}).get(cid, 0)
                rec = (leaders.get("levels", {}) or {}).get(child_lv, {}).get(cid, {}) or {}
                lead_id = rec.get("leader")
                din = rec.get("deg_in", np.nan)
                dgl = rec.get("deg_global", np.nan)
                lead_nm = nmap.get(lead_id, lead_id)
                z = (zmap.get(child_lv, {}) if zmap else {}).get(cid, None)
                z_for_label = z if (z is not None and np.isfinite(z)) else float("nan")
                topics_txt = (tmap.get(child_lv, {}) or {}).get(cid, "")

                label = label_template.format(
                    level=child_lv,
                    cid=cid,
                    leader=(lead_nm or "–"),
                    z=z_for_label,
                    n=sz,
                    deg_in=din,
                    deg_global=dgl,
                    topics=topics_txt,
                )
                tip = (
                    f"<b>{child_lv} {cid}</b><br/>Persons: {sz}<br/>Leader: {_esc(lead_nm) if lead_nm else '–'}"
                    f"{'<br/>z-score: %.1f' % z if (z is not None and np.isfinite(z)) else ''}"
                    f"<br/>k_in={'' if np.isnan(din) else int(din)} · k={'' if np.isnan(dgl) else int(dgl)}"
                    f"<br/>Topics: {_esc(topics_txt) if topics_txt else '–'}"
                )

                val = min(85, 12 + int(np.sqrt(max(sz, 1))))
                net.add_node(cnode, label=label, level=depth + 1, color=COLORS[child_lv], value=val, title=tip)
                net.add_edge(pnode, cnode)

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    net.write_html(out_html, notebook=False, open_browser=False)
    return out_html