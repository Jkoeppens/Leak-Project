"""
Orgchart renderer with topic overlays and per-cluster topic summaries (inline + optional pie tooltip).

Drop-in replacement / superset for prior `build_org_html`:

build_org_html(
    selection: dict,                 # hierarchical selection (levels -> clusters with children)
    leaders: dict | None,            # {(level, cid) -> {"leader": name, ...}} or similar
    H: "pd.DataFrame",               # nodes-to-clusters mapping with level_1..level_k columns
    persons: "pd.DataFrame | None" = None,
    out_html: str = "organigram_interaktiv.html",
    max_depth: int = 3,
    z_by_level: dict | None = None,  # {"L1": {cid: z, ...}, ...}
    topics_by_level: dict | None = None,  # {"L2": {cid: [(label, weight), ...]}, ...}
    physics: bool = False,
    label_template: str = "{level}:{cid} • {leader} • n={n} • z={z:.1f} • {topics}",
    label_size: int = 18,
    extra_edges: list | None = None, # [{level:"L2", src:cid, dst:cid, width:..., color:..., label:..., arrows:"to", dashes:True, smooth:True}, ...]
    E: "pd.DataFrame | None" = None, # optional original edges to compute deg stats (best-effort)
    pie_in_tooltip: bool = False,    # if True, render per-cluster topic mix as an inline base64 PNG pie chart
) -> str

Assumptions about `selection` structure (common in prior notebooks):
- selection is a dict keyed by levels like "L1", "L2", ... each containing a list of cluster dicts
- each cluster dict: {"cid": int, "keep": bool (optional, default True), "children": [child cluster dicts at next level], ...}

If your selection differs, you can adapt `_iter_selection()` accordingly.
"""
from __future__ import annotations

import io
import base64
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, List, Any

# Third-party
try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    pd = None

from pyvis.network import Network

# ------------------------- helpers -------------------------

def _esc(s: Any) -> str:
    if s is None:
        return ""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            )


def _topics_inline_summary(items: List[Tuple[str, float]] | None, k: int = 3) -> str:
    if not items:
        return "–"
    # normalize and take top-k
    s = float(sum(max(0.0, float(w)) for _, w in items)) or 1.0
    top = sorted(items, key=lambda t: t[1], reverse=True)[:k]
    return " / ".join(f"{lab} {int(100*float(w)/s + 0.5)}%" for lab, w in top)


def _pie_png_base64(items: List[Tuple[str, float]] | None, size_px: int = 120) -> str | None:
    if not items:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    vals = [max(0.0, float(w)) for _, w in items]
    if not any(v > 0 for v in vals):
        return None
    labels = [str(l) for l, _ in items]
    fig, ax = plt.subplots(figsize=(size_px/100.0, size_px/100.0), dpi=100)
    ax.pie(vals, labels=None, startangle=90)
    ax.axis('equal')
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------- selection traversal ---------------------

def _level_keys(selection: dict) -> List[str]:
    # Return sorted level keys as ["L1", "L2", ...]
    ks = [k for k in selection.keys() if k and (k[0].lower()=="l")]
    try:
        ks = sorted(ks, key=lambda k: int(str(k).lower().replace("l", "")))
    except Exception:
        ks = sorted(ks)
    return ks


def _iter_selection(selection: dict, max_depth: int | None = None) -> Iterable[Tuple[str, dict, str | None, int | None]]:
    """Yield (level_key, cluster_dict, parent_level_key, parent_cid) in hierarchical order.
    Expects each cluster_dict to possibly include 'children' list for the next level.
    """
    levels = _level_keys(selection)
    if not levels:
        return
    if max_depth is not None:
        levels = levels[:max_depth]

    # BFS over selection hierarchy
    parent_map = {None: selection.get(levels[0], [])}
    queue: List[Tuple[str, dict, str | None, int | None]] = []

    # seed with top level
    for c in selection.get(levels[0], []):
        queue.append((levels[0], c, None, None))

    i = 0
    while i < len(queue):
        lv, cdict, plv, pcid = queue[i]
        i += 1
        yield lv, cdict, plv, pcid
        # enqueue children if there's a deeper level
        lv_idx = levels.index(lv)
        if lv_idx + 1 < len(levels):
            child_lv = levels[lv_idx + 1]
            for ch in (cdict.get("children") or []):
                queue.append((child_lv, ch, lv, cdict.get("cid")))


# ---------------------- size & z lookups ----------------------

def _sizes_per_level(H: "pd.DataFrame") -> Dict[str, Dict[int, int]]:
    sizes: Dict[str, Dict[int, int]] = {}
    if H is None or pd is None:
        return sizes
    for col in [c for c in H.columns if c.lower().startswith("level_")]:
        lv = f"L{str(col).split('_')[1]}" if '_' in str(col) else str(col)
        try:
            gp = (H[[col]].dropna().astype({col: int}).value_counts().reset_index(name='n'))
            sizes[lv] = {int(r[col]): int(r['n']) for _, r in gp.iterrows()}
        except Exception:
            # robust fallback
            vc = H[col].dropna().astype(int).value_counts()
            sizes[lv] = {int(k): int(v) for k, v in vc.items()}
    return sizes


def _z_for(z_by_level: dict | None, lv: str, cid: int) -> float:
    if not z_by_level:
        return 0.0
    d = z_by_level.get(lv) or {}
    try:
        return float(d.get(int(cid), 0.0))
    except Exception:
        return 0.0


# --------------------------- main ---------------------------

def build_org_html(
    selection: dict,
    leaders: dict | None,
    H: "pd.DataFrame",
    persons: "pd.DataFrame | None" = None,
    out_html: str = "organigram_interaktiv.html",
    max_depth: int = 3,
    z_by_level: dict | None = None,
    topics_by_level: dict | None = None,
    physics: bool = False,
    label_template: str = "{level}:{cid} • {leader} • n={n} • z={z:.1f} • {topics}",
    label_size: int = 18,
    extra_edges: list | None = None,
    E: "pd.DataFrame | None" = None,
    pie_in_tooltip: bool = False,
) -> str:
    """Render an interactive org chart HTML with optional topic overlays and per-cluster topic summaries.

    Returns the absolute path of the written HTML file.
    """
    # Build a pyvis network with hierarchical layout
    net = Network(height="900px", width="100%", directed=True, notebook=False)
    net.barnes_hut()  # decent defaults; we'll disable physics per edge later

    # Global options for a tidy hierarchical layout
    import json

opts = {
  "layout": {
    "hierarchical": {
      "enabled": True,
      "direction": "UD",
      "sortMethod": "hubsize",
      "nodeSpacing": 180,
      "levelSeparation": 220,
    }
  },
  "physics": { "enabled": bool(physics) },
  "interaction": { "hover": True, "tooltipDelay": 120 }
}

net.set_options(json.dumps(opts))

    # Pre-compute sizes per level
    sizes = _sizes_per_level(H)

    # Index leaders for quick lookup
    def leader_name(lv: str, cid: int) -> str:
        if not leaders:
            return "–"
        key_variants = [
            (lv, int(cid)),
            (lv.lower(), int(cid)),
            (lv.upper(), int(cid)),
            (int(lv.replace('L','')) if str(lv).upper().startswith('L') and str(lv)[1:].isdigit() else lv, int(cid)),
        ]
        for k in key_variants:
            v = leaders.get(k)
            if isinstance(v, dict) and v.get("leader"):
                return str(v.get("leader"))
            if isinstance(v, str):
                return v
        return "–"

    # Add a root node to hang L1 clusters from (if not explicitly present)
    ROOT_ID = "EXEC"
    net.add_node(
        ROOT_ID,
        label="EXEC",
        shape="box",
        font={"size": label_size + 2, "face": "arial", "bold": True},
        color={"background": "#111827", "border": "#111827"},
        level=0,
    )

    # Track existing node ids for overlay validation
    existing_ids = set([ROOT_ID])

    # Build hierarchy nodes & edges
    # We expect selection as nested clusters across L1..Lk
    for lv, cdict, plv, pcid in _iter_selection(selection, max_depth=max_depth):
        cid = int(cdict.get("cid"))
        keep = cdict.get("keep", True)
        if not keep:
            continue
        node_id = f"{lv}:{cid}"
        n = sizes.get(lv, {}).get(cid, 0)
        z = _z_for(z_by_level, lv, cid)
        topics_list = None
        if topics_by_level and (lv in topics_by_level):
            topics_list = topics_by_level.get(lv, {}).get(cid)
        topics_inline = _topics_inline_summary(topics_list, k=3)
        lead_nm = leader_name(lv, cid)

        # label
        try:
            label = label_template.format(
                level=lv, cid=cid, leader=lead_nm or "–",
                n=n, z=z, deg_in=0, deg_global=0, topics=topics_inline
            )
        except Exception:
            label = f"{lv}:{cid} • {lead_nm or '–'} • n={n} • z={z:.1f}"

        # tooltip
        tip = f"<b>{_esc(lv)}:{cid}</b><br/>Leader: {_esc(lead_nm)}<br/>n: {n}<br/>z: {z:.2f}"
        if topics_list:
            s = float(sum(max(0.0, float(w)) for _, w in topics_list)) or 1.0
            top5 = sorted(topics_list, key=lambda t: t[1], reverse=True)[:5]
            tip += "<br/>Topics: " + ", ".join(f"{_esc(l)} ({int(100*float(w)/s+0.5)}%)" for l, w in top5)
            if pie_in_tooltip:
                img64 = _pie_png_base64(topics_list)
                if img64:
                    tip += f'<br/><img src="data:image/png;base64,{img64}" width="120" height="120" />'

        net.add_node(
            node_id,
            label=label,
            title=tip,
            shape="box",
            font={"size": label_size, "face": "arial"},
            color={"background": "#F9FAFB", "border": "#9CA3AF"},
            level=int(str(lv).lower().replace('l','')) if str(lv).lower().startswith('l') and str(lv)[1:].isdigit() else None,
        )
        existing_ids.add(node_id)

        # hierarchical edge from parent
        if pcid is None:
            net.add_edge(ROOT_ID, node_id, physics=False)
        else:
            parent_id = f"{plv}:{int(pcid)}"
            if parent_id in existing_ids:
                net.add_edge(parent_id, node_id, physics=False)
            else:
                # parent node not created (filtered out) => hang from root
                net.add_edge(ROOT_ID, node_id, physics=False)

    # Overlay topic flow edges (do not affect layout)
    if extra_edges:
        for e in extra_edges:
            lv = e.get("level")
            src_cid = e.get("src")
            dst_cid = e.get("dst")
            if lv is None or src_cid is None or dst_cid is None:
                continue
            src = f"{lv}:{int(src_cid)}"
            dst = f"{lv}:{int(dst_cid)}"
            if src not in existing_ids or dst not in existing_ids or src == dst:
                continue
            kwargs = {}
            if "width" in e:
                try:
                    kwargs["width"] = float(e["width"])
                except Exception:
                    pass
            if "color" in e:
                kwargs["color"] = str(e["color"])  # hex or color name
            if e.get("dashes", False):
                kwargs["dashes"] = True
            if e.get("smooth", True):
                kwargs["smooth"] = True
            if "label" in e:
                kwargs["label"] = str(e["label"])  # e.g., topic name or weight
            if "arrows" in e:
                kwargs["arrows"] = e["arrows"]  # e.g., 'to'
            kwargs["physics"] = False  # never influence layout
            net.add_edge(src, dst, **kwargs)

    # Write HTML
    net.write_html(out_html)
    return out_html
