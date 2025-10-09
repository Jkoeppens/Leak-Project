# === pipe/topics_flow/export_network_viz.py ===
import pandas as pd
import numpy as np
from pathlib import Path
from pyvis.network import Network
import datetime as dt


def export_network_viz(
    env,
    topic_filter=None,
    min_edge_weight=None,
    hide_self_loops=True,
    bg_color="#ffffff",
    max_width=8.0,
    width_quantile=0.9,
    height_px=900,
    hierarchical_direction="UD"
):
    """
    Exportiert ein interaktives vis-network (pyvis) HTML.
    - erkennt automatisch Spaltennamen in nodes_topics.csv / edges_topics.csv
    - nutzt hierarchische Struktur (":"-getrennte IDs)
    - ignoriert ungültige Edges oder fehlende Nodes
    - unterstützt Perioden-Suffixe (z. B. topic_flows_filtered_2001Q1.csv)
    """

    print("\n=== [export_network_viz] Start ===")

    org_dir = Path(env["outputs"]["org_dir"])
    viz_dir = org_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = viz_dir / "nodes_topics.csv"
    edges_path = viz_dir / "edges_topics.csv"
    assert nodes_path.exists(), f"❌ Datei fehlt: {nodes_path}"
    assert edges_path.exists(), f"❌ Datei fehlt: {edges_path}"

    print(f"[ok] Eingabedateien: {nodes_path.name}, {edges_path.name}")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    print(f"[load] nodes={len(nodes)} | edges={len(edges)}")

    # === Automatische Spaltenerkennung ===
    possible_node_cols = ["node_id", "id", "cluster_id"]
    node_id_col = next((c for c in possible_node_cols if c in nodes.columns), None)
    assert node_id_col, f"❌ Keine gültige ID-Spalte in nodes_topics.csv gefunden: {nodes.columns}"

    possible_src_cols = ["src", "source", "sender_cluster"]
    possible_dst_cols = ["dst", "target", "recipient_cluster"]
    src_col = next((c for c in possible_src_cols if c in edges.columns), None)
    dst_col = next((c for c in possible_dst_cols if c in edges.columns), None)
    assert src_col and dst_col, f"❌ Keine gültigen Quell-/Zielspalten in edges_topics.csv: {edges.columns}"

    weight_col = "weight" if "weight" in edges.columns else None
    print(f"[info] node_id_col={node_id_col} | src_col={src_col} | dst_col={dst_col} | weight_col={weight_col}")

    # === Netzwerk aufbauen ===
    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor=bg_color,
        directed=True,
        notebook=False
    )

    # Hierarchisches Layout aktivieren (aktuelle pyvis-Version)
    net.set_options(f"""
    var options = {{
      layout: {{
        hierarchical: {{
          direction: '{hierarchical_direction}',
          sortMethod: 'directed'
        }}
      }},
      physics: {{
        enabled: false
      }}
    }}
    """)

    # === Knoten hinzufügen ===
    for _, row in nodes.iterrows():
        node_id = str(row[node_id_col]).strip()
        group = row.get("group", "org_cluster")
        net.add_node(node_id, label=node_id, group=group)

    # === Kanten hinzufügen (mit Debug) ===
    valid_nodes = set(nodes[node_id_col].astype(str))
    added_edges = 0
    skipped_edges = []
    for _, e in edges.iterrows():
        src = str(e[src_col]).strip()
        dst = str(e[dst_col]).strip()
        if hide_self_loops and src == dst:
            skipped_edges.append((src, dst, "self-loop"))
            continue
        if src not in valid_nodes or dst not in valid_nodes:
            skipped_edges.append((src, dst, "missing node"))
            continue

        w = e.get(weight_col, 1.0)
        color = e.get("color", "#999999")
        net.add_edge(src, dst, value=float(w), color=color)
        added_edges += 1

    print(f"[done] Added {added_edges}/{len(edges)} edges")
    if skipped_edges:
        print(f"[debug] Skipped edges: {len(skipped_edges)}")
        print("Beispiele:")
        for s in skipped_edges[:10]:
            print("  ", s)

    # === Export ===
    out_html = viz_dir / f"org_topics_hier.html"
    net.save_graph(str(out_html))

    print(f"[write] {out_html}  | nodes={len(nodes)} edges={added_edges}")
    print("✅ [export_network_viz] fertig.")


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    export_network_viz(env)