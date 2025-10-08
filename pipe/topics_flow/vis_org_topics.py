"""
pipe.vis_org_topics
===================
Erzeugt ein interaktives Hierarchie-Netzwerk (Organigramm + Topic-Layer)
als vis-network/pyvis-HTML. Knoten sind Cluster (Abteilungen),
deren Pies thematische Anteile zeigen; Kanten sind farblich nach Topic kodiert.

Inputs:
  - reports/orgchart/viz/nodes_topics.csv
  - reports/orgchart/viz/edges_topics.csv
  - reports/orgchart/viz/pies/*.png

Output:
  - reports/orgchart/viz/org_topics_hier.html
"""

from pathlib import Path
import pandas as pd
import numpy as np
from pyvis.network import Network


def vis_org_topics(env: dict) -> None:
    org_dir = Path(env["outputs"]["org_dir"])

    VIZ_DIR   = org_dir / "viz"
    PIE_DIR   = VIZ_DIR / "pies"
    NODES_CSV = VIZ_DIR / "nodes_topics.csv"
    EDGES_CSV = VIZ_DIR / "edges_topics.csv"
    HTML_OUT  = VIZ_DIR / "org_topics_hier.html"

    # ---- Anzeigeparameter ----
    TOPIC_FILTER    = None      # z.B. 30
    MIN_EDGE_W      = None      # z.B. 10000
    HIDE_SELF_LOOPS = True
    WIDTH_QUANTILE  = 0.90
    MAX_WIDTH       = 8.0
    BG_COLOR        = "#ffffff"

    # ---- Laden ----
    nodes = pd.read_csv(NODES_CSV)
    edges = pd.read_csv(EDGES_CSV)

    # ---- Filter ----
    if TOPIC_FILTER is not None:
        edges = edges[edges["topic_id"] == TOPIC_FILTER]
    if MIN_EDGE_W is not None:
        edges = edges[edges["weight"] >= MIN_EDGE_W]
    if HIDE_SELF_LOOPS:
        edges = edges[edges["source"] != edges["target"]].copy()

    # ---- Kantenskalierung ----
    if len(edges):
        cap = float(edges["weight"].quantile(WIDTH_QUANTILE))
        if cap <= 0:
            cap = float(edges["weight"].max())
        edges["width"] = 1.0 + (np.minimum(edges["weight"], cap) / max(cap, 1.0)) * (MAX_WIDTH - 1.0)
    else:
        edges["width"] = 1.0

    # ---- Hierarchieebene aus cluster_id ----
    def level_from_cluster_id(cid: str) -> int:
        if pd.isna(cid): return 0
        return len(str(cid).split(":"))

    nodes["id"] = nodes["id"].astype(str)
    nodes["level"] = nodes["id"].map(level_from_cluster_id)

    # ---- Tooltip ----
    def fmt_top_topics(s): return s if isinstance(s, str) else ""

    nodes["title"] = nodes.apply(
        lambda r: f"<b>{r['id']}</b><br/>total={int(r.get('node_weight_total',0))}<br/>{fmt_top_topics(r.get('top_topics',''))}",
        axis=1
    )

    def pie_path_for(nid: str) -> Path:
        return PIE_DIR / f"{nid.replace(':','_')}.png"

    nodes["has_pie"] = nodes["id"].apply(lambda x: pie_path_for(x).exists())

    # ---- Netzwerk ----
    net = Network(height="900px", width="100%", directed=True, cdn_resources="in_line", bgcolor=BG_COLOR)
    net.set_options("""
    {
      "nodes": {
        "shape": "image",
        "size": 26,
        "borderWidth": 1
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
        "smooth": {"type": "dynamic"}
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD",
          "sortMethod": "directed",
          "nodeSpacing": 150,
          "levelSeparation": 120,
          "treeSpacing": 200
        }
      },
      "physics": {"enabled": false},
      "interaction": {"hover": true, "tooltipDelay": 120}
    }
    """)

    # ---- Nodes ----
    for _, r in nodes.iterrows():
        nid = str(r["id"])
        kwargs = dict(title=r["title"], level=int(r.get("level", 0)))
        pth = pie_path_for(nid)
        if r.get("has_pie", False):
            kwargs.update(dict(image=str(pth), shape="image"))
        else:
            kwargs.update(dict(shape="dot", color="#bbbbbb", label=nid))
        net.add_node(nid, **kwargs)

    # ---- Edges ----
    node_ids = set(nodes["id"].astype(str))
    for _, e in edges.iterrows():
        src, dst = str(e["source"]), str(e["target"])
        if src not in node_ids or dst not in node_ids:
            continue
        net.add_edge(
            src, dst,
            color=e.get("color", "#888"),
            width=float(e.get("width", 1.0)),
            title=f"topic={e.get('topic_id')} | weight={int(e.get('weight',0))} | n={int(e.get('n_events',0))}"
        )

    # ---- Export ----
    net.show(str(HTML_OUT))
    print(f"[write] {HTML_OUT}")
    print(f"Nodes: {len(nodes)} | Edges (shown): {len(edges)} | topic_filter={TOPIC_FILTER}")


if __name__ == "__main__":
    from pipe.setup_env import setup_environment
    env = setup_environment()
    vis_org_topics(env)