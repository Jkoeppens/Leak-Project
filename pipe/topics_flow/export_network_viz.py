# === export_network_viz_colab.py ===
import os
import json
import pandas as pd
from pathlib import Path
from pyvis.network import Network
from IPython.display import IFrame, display

def export_network_viz_colab(
    base_dir="/content",
    topic_filter=None,
    min_edge_weight=None,
    hide_self_loops=True,
    bg_color="#ffffff",
    max_width=8.0,
    width_quantile=0.9,
    height_px=900,
    hierarchical_direction="UD",
    debug=False
):
    """
    Exportiert ein interaktives pyvis-Netzwerk (vis.js) im Colab-Kontext.
    Voraussetzungen:
      - Dateien: /content/viz/nodes_topics.csv, edges_topics.csv, optional cluster_labels.csv
      - erzeugt: /content/viz/org_topics_hier.html
    """

    print("\n=== [export_network_viz_colab] Start ===")

    # --- Pfade ---
    viz_dir = Path(base_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = viz_dir / "nodes_topics.csv"
    edges_path = viz_dir / "edges_topics.csv"
    labels_path = viz_dir / "cluster_labels.csv"

    assert nodes_path.exists(), f"❌ Datei fehlt: {nodes_path}"
    assert edges_path.exists(), f"❌ Datei fehlt: {edges_path}"
    print(f"[ok] Eingabedateien: {nodes_path.name}, {edges_path.name}")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    print(f"[load] nodes={len(nodes)} | edges={len(edges)}")

    # --- Labels laden ---
    label_map = {}
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        id_col = next((c for c in ["id", "cluster_id", "node_id"] if c in labels_df.columns), None)
        label_col = next((c for c in ["label", "topic_label"] if c in labels_df.columns), None)
        if id_col and label_col:
            label_map = dict(zip(labels_df[id_col].astype(str), labels_df[label_col]))
            print(f"[labels] Loaded {len(label_map)} labels (columns: {id_col}, {label_col})")
        else:
            print(f"[warn] Keine passenden Spalten in cluster_labels.csv: {labels_df.columns.tolist()}")
    else:
        print(f"[warn] cluster_labels.csv nicht gefunden unter {labels_path}")

    # --- Hilfsfunktion: hierarchische Labelsuche ---
    def find_hierarchical_label(node_id: str):
        parts = node_id.split(":")
        for i in range(len(parts), 0, -1):
            prefix = ":".join(parts[:i])
            if prefix in label_map:
                return label_map[prefix]
        return None

    # --- Automatische Spaltenerkennung ---
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

    # --- Netzwerk aufbauen ---
    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor=bg_color,
        directed=True,
        notebook=False
    )

    options = {
        "layout": {"hierarchical": {"direction": hierarchical_direction, "sortMethod": "directed"}},
        "physics": {"enabled": False}
    }
    net.set_options(json.dumps(options))

    # --- Pies laden ---
    pie_dir = viz_dir / "pies"
    pie_index_path = pie_dir / "cluster_pies_index.csv"
    pie_map = {}
    if pie_index_path.exists():
        pie_index = pd.read_csv(pie_index_path)
        pie_map = dict(zip(pie_index["cluster_id"].astype(str), pie_index["pie_path"]))
        print(f"[viz] {len(pie_map)} Pie-Images geladen aus {pie_index_path.name}")
    else:
        print("[viz] Keine Pie-Images gefunden (pies/cluster_pies_index.csv fehlt)")

    # --- Knoten hinzufügen ---
    valid_nodes = set(nodes[node_id_col].astype(str))
    exact_match = hierarchical_match = with_image = 0

    for _, row in nodes.iterrows():
        node_id = str(row[node_id_col]).strip()
        group = row.get("group", "org_cluster")

        label = label_map.get(node_id)
        if label is not None:
            exact_match += 1
        else:
            label = find_hierarchical_label(node_id)
            if label is not None:
                hierarchical_match += 1

        label = label or node_id
        short_label = (label[:80] + "…") if len(label) > 80 else label

        img_rel = None
        if node_id in pie_map:
            img_rel = os.path.relpath(pie_dir / pie_map[node_id], viz_dir)

        if img_rel:
            net.add_node(
                node_id,
                label=short_label,
                title=label,
                shape="image",
                image=img_rel
            )
            with_image += 1
        else:
            net.add_node(node_id, label=short_label, title=label, group=group)

    print(f"[label-stats] exact={exact_match} | hierarchical={hierarchical_match} | with_image={with_image} | total={len(nodes)}")

    # --- Kanten hinzufügen ---
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

    if debug:
        print(f"[debug] Added {added_edges}/{len(edges)} edges")
        if skipped_edges:
            print(f"[debug] Skipped {len(skipped_edges)} edges (first 10):")
            for s in skipped_edges[:10]:
                print("  ", s)

    # --- Export ---
    out_html = viz_dir / "org_topics_hier.html"
    net.save_graph(str(out_html))
    print(f"[write] {out_html}  | nodes={len(nodes)} edges={added_edges}")
    print("✅ [export_network_viz_colab] fertig.")

    # --- Anzeige / Download ---
    display(IFrame(src=str(out_html), width="100%", height=height_px))
    print(f"💾 Download-Link: {out_html}")

# Beispielaufruf:
# (nachdem du deine CSV-Dateien hochgeladen hast)
# from google.colab import files
# files.upload()  # Lade nodes_topics.csv, edges_topics.csv etc. hoch
#
# export_network_viz_colab(debug=True)
