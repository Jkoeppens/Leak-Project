"""
Orgchart renderer with topic overlays and per-cluster topic summaries.

Public API:
    build_org_html(
        selection: dict,                 # {"L1":[{cid, children:[...]}], "L2":[...], ...}
        leaders: dict | None,            # {(lv, cid)->"Name"} oder {(lv,cid)->{"leader":...}}
        H: "pd.DataFrame",               # DataFrame mit level_1..level_k Spalten (für n je Cluster)
        persons: "pd.DataFrame | None" = None,
        out_html: str = "organigram_interaktiv.html",
        max_depth: int = 5,
        z_by_level: dict | None = None,  # {"Lx": {cid->z}}
        topics_by_level: dict | None = None,  # {"Lx": {cid: [(label, weight), ...]}, ...}
        physics: bool = False,
        label_template: str = "{level}:{cid} • n={n} • {topics}",
        label_size: int = 16,
        extra_edges: list | None = None, # [{level:"L2", src:cid, dst:cid, width:..., color:..., label:..., arrows:"to", dashes:True, smooth:True}, ...]
        E: "pd.DataFrame | None" = None,
        pie_in_tooltip: bool = True,
        debug: bool = True
    ) -> str
"""

from __future__ import annotations

import io
import json
import base64
from typing import Dict, Tuple, Iterable, List, Any

# Third-party (optional pandas)
try:
    import pandas as pd
except Exception:
    pd = None

from pyvis.network import Network


# --------------------- small utils ---------------------

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
    s = float(sum(max(0.0, float(w)) for _, w in items)) or 1.0
    top = sorted(items, key=lambda t: float(t[1]), reverse=True)[:k]
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
    fig, ax = plt.subplots(figsize=(size_px/100.0, size_px/100.0), dpi=100)
    ax.pie(vals, labels=None, startangle=90)
    ax.axis("equal")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _lv_key_from_str_cluster(cid: Any) -> str:
    s = str(cid)
    return f"L{s.count(':')+1}"


# --------------------- selection traversal ---------------------

def _level_keys(selection: dict) -> List[str]:
    ks = [k for k in selection.keys() if k and (str(k)[0].lower() == "l")]
    try:
        ks = sorted(ks, key=lambda k: int(str(k).lower().replace("l", "")))
    except Exception:
        ks = sorted(ks)
    return ks

def _iter_selection(selection: dict, max_depth: int | None = None) -> Iterable[Tuple[str, dict, str | None, Any | None]]:
    """Yield (level_key, cluster_dict, parent_level_key, parent_cid) in hierarchical order (BFS)."""
    levels = _level_keys(selection)
    if not levels:
        return
    if max_depth is not None:
        levels = levels[:max_depth]

    queue: List[Tuple[str, dict, str | None, Any | None]] = []
    for c in selection.get(levels[0], []):
        queue.append((levels[0], c, None, None))

    i = 0
    while i < len(queue):
        lv, cdict, plv, pcid = queue[i]
        i += 1
        yield lv, cdict, plv, pcid
        lv_idx = levels.index(lv)
        if lv_idx + 1 < len(levels):
            child_lv = levels[lv_idx + 1]
            for ch in (cdict.get("children") or []):
                queue.append((child_lv, ch, lv, cdict.get("cid")))


# --------------------- sizes & z lookups ---------------------

def _sizes_per_level(H: "pd.DataFrame") -> Dict[str, Dict[Any, int]]:
    sizes: Dict[str, Dict[Any, int]] = {}
    if H is None or pd is None:
        return sizes
    cols = [c for c in H.columns if str(c).lower().startswith("level_")]
    for col in cols:
        lv = f"L{str(col).split('_')[1]}" if '_' in str(col) else str(col)
        try:
            vc = H[col].dropna().astype(object).value_counts()
        except Exception:
            vc = H[col].dropna().value_counts()
        sizes[lv] = {k: int(v) for k, v in vc.items()}
    return sizes

def _z_for(z_by_level: dict | None, lv: str, cid: Any) -> float:
    if not z_by_level:
        return 0.0
    d = z_by_level.get(lv) or {}
    try:
        return float(d.get(cid, d.get(int(cid), 0.0)))
    except Exception:
        return float(d.get(cid, 0.0))


# --------------------- safe set_options ---------------------

def _safe_set_options(net: Network, opts: dict, debug_path: str | None = None) -> None:
    """Serialize dict to JSON and call set_options. If anything fails, fallback to a minimal valid config."""
    try:
        net.set_options(json.dumps(opts))
    except Exception as e:
        if debug_path:
            try:
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(opts, f, indent=2)
            except Exception:
                pass
        fallback = {
            "layout": {"hierarchical": {"enabled": True, "direction": "UD"}},
            "physics": {"enabled": False},
            "interaction": {"hover": True, "tooltipDelay": 120},
        }
        net.set_options(json.dumps(fallback))


# --------------------------- main API ---------------------------

def build_org_html(
    selection: dict,
    leaders: dict | None,
    H: "pd.DataFrame",
    persons: "pd.DataFrame | None" = None,
    out_html: str = "organigram_interaktiv.html",
    max_depth: int = 5,
    z_by_level: dict | None = None,
    topics_by_level: dict | None = None,
    physics: bool = False,
    label_template: str = "{level}:{cid} • n={n} • {topics}",
    label_size: int = 16,
    extra_edges: list | None = None,
    E: "pd.DataFrame | None" = None,
    pie_in_tooltip: bool = True,
    debug: bool = True,
) -> str:
    """Render an interactive org chart HTML with optional topic overlays and per-cluster topic summaries.
    Returns absolute path to the written HTML file.
    """
    net = Network(height="900px", width="100%", directed=True, notebook=False)
    net.barnes_hut()  # sensible defaults

    # always feed valid JSON
    opts = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "hubsize",
                "nodeSpacing": 180,
                "levelSeparation": 220
            }
        },
        "physics": {"enabled": bool(physics)},
        "interaction": {"hover": True, "tooltipDelay": 120}
    }
    _safe_set_options(net, opts, "vis_options_debug.json" if debug else None)

    sizes = _sizes_per_level(H)

    def leader_name(lv: str, cid: Any) -> str:
        if not leaders:
            return "–"
        variants = [
            (lv, cid),
            (lv.lower(), cid),
            (lv.upper(), cid),
        ]
        try:
            variants += [
                (int(lv.replace("L","")), int(cid))  # (2, 143)
            ]
        except Exception:
            pass
        for k in variants:
            v = leaders.get(k)
            if isinstance(v, dict) and v.get("leader"):
                return str(v.get("leader"))
            if isinstance(v, str):
                return v
        return "–"

    # Root
    ROOT_ID = "EXEC"
    net.add_node(
        ROOT_ID,
        label="EXEC",
        shape="box",
        font={"size": label_size + 2, "face": "arial", "bold": True},
        color={"background": "#111827", "border": "#111827"},
        level=0,
    )
    existing_ids = {ROOT_ID}

    # Nodes & tree edges
    n_nodes = 0
    n_tree_edges = 0
    for lv, cdict, plv, pcid in _iter_selection(selection, max_depth=max_depth):
        if cdict is None:
            continue
        cid = cdict.get("cid")
        keep = cdict.get("keep", True)
        if not keep:
            continue
        node_id = f"{lv}:{cid}"
        n_val = sizes.get(lv, {}).get(cid, 0)
        z_val = _z_for(z_by_level, lv, cid)
        t_list = topics_by_level.get(lv, {}).get(cid) if topics_by_level else None
        topics_inline = _topics_inline_summary(t_list, k=3)
        lead_nm = leader_name(lv, cid)

        # label
        try:
            label = label_template.format(level=lv, cid=cid, n=n_val, z=z_val, topics=topics_inline)
        except Exception:
            label = f"{lv}:{cid} • n={n_val} • {topics_inline}"

        # tooltip
        tip = f"<b>{_esc(lv)}:{_esc(cid)}</b><br/>Leader: {_esc(lead_nm)}<br/>n: {n_val}<br/>z: {z_val:.2f}"
        if t_list:
            s = float(sum(max(0.0, float(w)) for _, w in t_list)) or 1.0
            top5 = sorted(t_list, key=lambda t: float(t[1]), reverse=True)[:5]
            tip += "<br/>Topics: " + ", ".join(f"{_esc(l)} ({int(100*float(w)/s+0.5)}%)" for l, w in top5)
            if pie_in_tooltip:
                img64 = _pie_png_base64(t_list)
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
        n_nodes += 1

        # parent edge
        if pcid is None:
            net.add_edge(ROOT_ID, node_id, physics=False)
            n_tree_edges += 1
        else:
            parent_id = f"{plv}:{pcid}"
            if parent_id in existing_ids:
                net.add_edge(parent_id, node_id, physics=False)
                n_tree_edges += 1
            else:
                # If parent not kept, hang from root to keep hierarchy layout stable
                net.add_edge(ROOT_ID, node_id, physics=False)
                n_tree_edges += 1

    # Topic overlay edges
    n_topic_edges = 0
    if extra_edges:
        for e in extra_edges:
            lv = e.get("level")
            src_cid = e.get("src")
            dst_cid = e.get("dst")
            if lv is None or src_cid is None or dst_cid is None:
                continue
            src = f"{lv}:{src_cid}"
            dst = f"{lv}:{dst_cid}"
            if src == dst:
                continue
            if (src not in existing_ids) or (dst not in existing_ids):
                # overlay nur zwischen existierenden Knoten
                continue
            kwargs = {"physics": False}
            # width/color/label etc.
            if "width" in e:
                try:
                    kwargs["width"] = float(e["width"])
                except Exception:
                    pass
            if "color" in e:
                kwargs["color"] = str(e["color"])
            if e.get("dashes", False):
                kwargs["dashes"] = True
            if e.get("smooth", True):
                kwargs["smooth"] = True
            if "label" in e:
                kwargs["label"] = str(e["label"])
            if "arrows" in e:
                kwargs["arrows"] = e["arrows"]
            net.add_edge(src, dst, **kwargs)
            n_topic_edges += 1

    if debug:
        print(f"[orgchart] nodes={n_nodes}  tree_edges={n_tree_edges}  topic_edges={n_topic_edges}")

    net.write_html(out_html)
    return out_html