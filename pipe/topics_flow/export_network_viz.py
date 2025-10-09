# === pipe/topics_flow/export_network_viz.py ===
"""
Exportiert ein interaktives vis-network (pyvis) HTML:
- Hierarchische Anordnung über ":"-Tiefe in cluster_id
- Pies-Icons, falls vorhanden (viz/pies/<cluster>.png)
- Robuste Behandlung fehlender Nodes (werden aus Edge-Endpunkten ergänzt)
- Optionale Filter: topic_id, min_edge_weight, self-loops
- Periodenerkennung, wenn periodische Artefakte existieren

Erwartete Inputs (typisch aus prepare/export-Schritten):
  - {org_dir}/viz/nodes_topics.csv
  - {org_dir}/viz/edges_topics.csv
  - {org_dir}/viz/pies/*.png (optional)

Outputs:
  - {org_dir}/viz/org_topics_hier[_{period}].html
"""

from pathlib import Path
import pandas as pd
import numpy as np
from pyvis.network import Network
import datetime as dt


def _period_from_existing(org_dir: Path) -> str | None:
    """
    Versucht eine Periodenkennung zu extrahieren, falls periodische Pies/Flows existieren.
    Z.B. cluster_topic_pies_summary_YYYY-MM.csv  -> 'YYYY-MM'
    Gibt None zurück, wenn nichts gefunden.
    """
    cands = sorted(org_dir.glob("cluster_topic_pies_summary_*.csv"))
    if not cands:
        return None
    stem = cands[-1].stem  # zuletzt (alphabetisch) -> i.d.R. neueste Periode
    parts = stem.split("_")
    if len(parts) >= 5:
        # cluster_topic_pies_summary_YYYY-MM
        return parts[-1]
    return None


def _safe_cols(df: pd.DataFrame, needed: list[str], name: str):
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: Spalten fehlen: {miss}. Vorhanden: {list(df.columns)[:10]}...")


def _hier_level(cid: str) -> int:
    """Anzahl ':' + 1, nichtnegativ."""
    if not isinstance(cid, str):
        cid = str(cid)
    return cid.count(":") + 1 if cid else 1


def _pie_path(pies_dir: Path, cid: str) -> Path:
    return pies_dir / f"{cid.replace(':','_')}.png"


def export_network_viz(
    env: dict,
    topic_filter: int | float | None = None,
    min_edge_weight: float | None = None,
    hide_self_loops: bool = True,
    bg_color: str = "#ffffff",
    max_width: float = 8.0,
    width_quantile: float = 0.90,
    height_px: int = 900,
    hierarchical_direction: str = "UD",
):
    print("\n=== [export_network_viz] Starte ===")

    # --- Pfade aus env ---
    org_dir = Path(env["outputs"]["org_dir"])
    viz_dir = Path(env["outputs"].get("viz_dir", org_dir / "viz"))
    viz_dir.mkdir(parents=True, exist_ok=True)
    pies_dir = viz_dir / "pies"

    # --- periodische Kennung (optional) ---
    period = _period_from_existing(org_dir)
    suffix = f"_{period}" if period else ""

    # --- Dateien: primär periodlos; wenn periodische Varianten existieren, werden sie vorher erzeugt
    nodes_csv = viz_dir / "nodes_topics.csv"
    edges_csv = viz_dir / "edges_topics.csv"

    if not nodes_csv.exists() or not edges_csv.exists():
        raise FileNotFoundError(
            f"Erwarte nodes/edges in {viz_dir}. Fehlend: "
            f"{'nodes_topics.csv' if not nodes_csv.exists() else ''} "
            f"{'edges_topics.csv' if not edges_csv.exists() else ''}"
        )

    # --- Laden ---
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    print(f"[load] nodes={len(nodes)} edges={len(edges)}")

    # --- Pflichtspalten prüfen & normalisieren ---
    # Nodes: brauchen 'id' zwingend
    _safe_cols(nodes, ["id"], "nodes_topics.csv")
    nodes["id"] = nodes["id"].astype(str)

    # Edges: erlauben 'source/target' (neu) oder 'src/dst' (legacy)
    edge_src = "source" if "source" in edges.columns else ("src" if "src" in edges.columns else None)
    edge_dst = "target" if "target" in edges.columns else ("dst" if "dst" in edges.columns else None)
    if not edge_src or not edge_dst:
        raise ValueError(f"edges_topics.csv: brauche 'source/target' oder 'src/dst'. Spalten: {list(edges.columns)}")

    # Minimalset
    need_e = [edge_src, edge_dst, "weight"]
    for c in need_e:
        if c not in edges.columns:
            raise ValueError(f"edges_topics.csv: Spalte fehlt: {c}")

    # --- Filter anwenden ---
    f = edges.copy()
    # Topic-Filter: topic_id oder (legacy) topic
    if topic_filter is not None:
        topic_col = "topic_id" if "topic_id" in f.columns else ("topic" if "topic" in f.columns else None)
        if topic_col:
            f = f[f[topic_col] == topic_filter]

    # min_edge_weight
    if min_edge_weight is not None:
        f = f[f["weight"] >= float(min_edge_weight)]

    # Self loops
    if hide_self_loops:
        f = f[f[edge_src] != f[edge_dst]]

    # --- Sicherstellen: alle Edge-Endpunkte als Nodes vorhanden ---
    node_ids = set(nodes["id"].astype(str))
    needed = set(f[edge_src].astype(str)).union(set(f[edge_dst].astype(str)))
    missing = list(needed - node_ids)

    if missing:
        print(f"[info] {len(missing)} fehlende Nodes werden placeholder-mäßig ergänzt.")
        placeholder = pd.DataFrame({"id": [str(x) for x in missing]})
        # sinnvolle Default-Metadaten
        placeholder["node_weight_total"] = np.nan
        placeholder["n_topics_nonzero"] = 0
        placeholder["top_topics"] = ""
        nodes = pd.concat([nodes, placeholder], ignore_index=True)

    # --- Kantenbreiten sanft skalieren (falls nicht vorhanden) ---
    if "width" not in f.columns or f["width"].isna().all():
        if len(f) > 0:
            cap = float(f["weight"].quantile(width_quantile))
            cap = cap if cap > 0 else float(f["weight"].max())
            f["width"] = 1.0 + (np.minimum(f["weight"], cap) / max(cap, 1.0)) * (max_width - 1.0)
        else:
            f["width"] = 1.0

    # --- PyVis initialisieren ---
    net = Network(
        height=f"{height_px}px",
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
        bgcolor=bg_color,
    )

    # Layout/Interaktion
    net.set_options(f"""
    {{
      "nodes": {{"borderWidth": 1}},
      "edges": {{
        "arrows": {{"to": {{"enabled": true, "scaleFactor": 0.8}}}},
        "smooth": {{"type": "dynamic"}}
      }},
      "layout": {{
        "hierarchical": {{
          "enabled": true,
          "direction": "{hierarchical_direction}",
          "sortMethod": "hubsize",
          "nodeSpacing": 160,
          "levelSeparation": 200
        }}
      }},
      "physics": {{"enabled": false}},
      "interaction": {{
        "hover": true,
        "tooltipDelay": 120,
        "hideEdgesOnDrag": false,
        "hideNodesOnDrag": false
      }}
    }}
    """)

    # --- Nodes hinzufügen ---
    # Pie-Icons, falls vorhanden; sonst als Box mit Label
    def node_title(row):
        tt = str(row.get("top_topics", "")) if not pd.isna(row.get("top_topics", "")) else ""
        tot = row.get("node_weight_total", np.nan)
        tot_txt = "–" if pd.isna(tot) else f"{int(tot)}"
        return f"<b>{row['id']}</b><br/>total={tot_txt}<br/>{tt}"

    # Level (Tiefe) aus ':' ableiten
    def node_level(cid: str) -> int:
        try:
            return int(_hier_level(cid))
        except Exception:
            return 1

    for _, r in nodes.iterrows():
        nid = str(r["id"])
        title = node_title(r)
        level = node_level(nid)

        # Pie?
        img = _pie_path(pies_dir, nid)
        if img.exists():
            net.add_node(
                nid,
                title=title,
                level=level,
                shape="image",
                image=str(img),
            )
        else:
            net.add_node(
                nid,
                title=title,
                level=level,
                label=nid,
                shape="box",
                color={"background": "#F9FAFB", "border": "#9CA3AF"},
                font={"size": 14, "face": "arial"},
            )

    # --- Edges hinzufügen ---
    # Sanity: unbekannte Endpunkte (sollten durch Placeholder bereits abgedeckt sein)
    missing_sources = [x for x in f[edge_src].astype(str).unique() if x not in net.get_nodes()]
    missing_targets = [x for x in f[edge_dst].astype(str).unique() if x not in net.get_nodes()]
    if missing_sources or missing_targets:
        print(f"[warn] Endpunkte fehlen noch immer im Netz: "
              f"src={len(missing_sources)} dst={len(missing_targets)} (werden übersprungen)")

    added = 0
    for _, e in f.iterrows():
        s = str(e[edge_src]); t = str(e[edge_dst])
        if (s not in net.get_nodes()) or (t not in net.get_nodes()) or s == t:
            continue
        color = e.get("color", "#888888")
        width = float(e.get("width", 1.0))
        topic_id = e.get("topic_id", e.get("topic", None))
        n = int(e.get("n_events", 0)) if not pd.isna(e.get("n_events", np.nan)) else 0
        val = float(e.get("weight", 0.0))
        label = f"topic={topic_id} • n={n} • w={int(val)}" if topic_id is not None else f"n={n} • w={int(val)}"

        net.add_edge(
            s, t,
            color=color,
            width=width,
            label=label,
            arrows="to",
            smooth=True,
            physics=False,
        )
        added += 1

    # --- Export ---
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_name = f"org_topics_hier{suffix}.html"
    out_path = viz_dir / out_name
    net.write_html(str(out_path))
    print(f"[write] {out_path}  | nodes={len(net.get_nodes())} edges={added}")
    print("✅ [export_network_viz] fertig.")


# --------------------------------------------------------------------
if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    export_network_viz(env)