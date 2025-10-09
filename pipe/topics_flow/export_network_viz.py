# === pipe/topics_flow/export_network_viz.py ===
import pandas as pd
from pathlib import Path
from pyvis.network import Network
import time

print("\n=== [DEBUG edges] ===")
    print("Columns:", list(edges.columns))
    print("Beispiel (erste 3 Zeilen):")
    print(edges.head(3).to_string(index=False))

    edges[src_col] = edges[src_col].astype(str)
    edges[dst_col] = edges[dst_col].astype(str)
    print(f"[info] src_col={src_col} | dst_col={dst_col} | topic_col={topic_col}")
    print(f"[info] edges rows={len(edges)}")

def export_network_viz(env: dict, period: str | None = None) -> None:
    """
    Erstellt interaktive HTML-Netzwerkvisualisierung (PyVis)
    aus vorbereiteten Topics-, Node- und Edge-Dateien.

    Erwartet:
      - nodes_topics.csv
      - edges_topics.csv
      - topic_colors.csv
    Optional:
      - period (z. B. '2001Q1') → verwendet periodenspezifische Dateien
    """

    print("\n=== [export_network_viz] Start ===")

    # === Pfade aus env ===
    org_dir = Path(env["outputs"]["org_dir"])
    viz_dir = Path(env["outputs"].get("viz_dir", org_dir / "viz"))
    viz_dir.mkdir(parents=True, exist_ok=True)

    # === Perioden-Erweiterung ===
    suf = f"_{period}" if period else ""
    nodes_path = viz_dir / f"nodes_topics{suf}.csv"
    edges_path = viz_dir / f"edges_topics{suf}.csv"
    colors_path = viz_dir / "topic_colors.csv"

    # === Fallback, falls periodische Dateien fehlen ===
    if not nodes_path.exists():
        nodes_path = viz_dir / "nodes_topics.csv"
    if not edges_path.exists():
        edges_path = viz_dir / "edges_topics.csv"

    for p in [nodes_path, edges_path]:
        assert p.exists(), f"❌ Datei fehlt: {p}"
    print(f"[ok] Eingabedateien: {nodes_path.name}, {edges_path.name}")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    colors = pd.read_csv(colors_path) if colors_path.exists() else pd.DataFrame()

    print(f"[load] nodes={len(nodes)} | edges={len(edges)}")

    # === Schema-Erkennung ===
    src_col = "source" if "source" in edges.columns else "sender_cluster"
    dst_col = "target" if "target" in edges.columns else "recipient_cluster"
    topic_col = "topic_id" if "topic_id" in edges.columns else "thread_topic_id"

    # === Fehlende Nodes ergänzen (damit keine Assertion mehr fliegt)
    node_ids = set(nodes["node_id"].astype(str))
    edge_nodes = set(edges[src_col].astype(str)) | set(edges[dst_col].astype(str))
    missing = edge_nodes - node_ids
    if missing:
        print(f"⚠️ Ergänze {len(missing)} fehlende Nodes (z. B. aus tieferer Hierarchie)")
        new_nodes = pd.DataFrame({
            "node_id": list(missing),
            "label": list(missing),
            "level": "Lx",
            "topics": "–"
        })
        nodes = pd.concat([nodes, new_nodes], ignore_index=True)

    # === PyVis Setup ===
    net = Network(height="900px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.set_options("""
    {
      "layout": { "hierarchical": { "enabled": true, "direction": "UD",
                                    "sortMethod": "hubsize", "nodeSpacing": 180,
                                    "levelSeparation": 220 } },
      "physics": { "enabled": false },
      "interaction": { "hover": true, "tooltipDelay": 120 }
    }
    """)

    # === Nodes einfügen ===
    for _, n in nodes.iterrows():
        label = n.get("label", n["node_id"])
        color = "#F9FAFB"
        border = "#9CA3AF"
        if isinstance(n.get("topics"), str) and len(n["topics"]) > 1:
            label += f"\n{n['topics']}"
        net.add_node(
            n["node_id"], label=label, shape="box",
            color={"background": color, "border": border},
            font={"size": 16, "face": "arial"}
        )

    # === Edges hinzufügen ===
    for _, e in edges.iterrows():
        src = str(e[src_col])
        dst = str(e[dst_col])
        if src == dst:
            arrows = "to"
        else:
            arrows = "to"

        col = e.get("color", "#666")
        width = float(e.get("width", 1.0))
        label = str(e.get(topic_col, ""))

        net.add_edge(src, dst, color=col, width=width, label=label, arrows=arrows, smooth=True, physics=False)

    # === Export ===
    ts = int(time.time())
    html_name = f"org_topics_hier{('_' + period) if period else ''}_inline.html"
    out_html = viz_dir / html_name
    net.write_html(str(out_html))

    print(f"[done] {html_name} | nodes={len(nodes)} | edges={len(edges)} | missing={len(missing)}")
    print(f"[open] {out_html}")


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    export_network_viz(env)