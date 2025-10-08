# === pipe/topics_flow/prepare_topic_colors_and_edges.py ===
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def prepare_topic_colors_and_edges(env):
    """
    Erzeugt konsistente Farbcodes pro Topic und extrahiert Edges für visuelle Netzwerke.
    Erwartet:
      - cluster_topic_pies_summary.csv (aus org_dir)
      - topic_flows_filtered.csv (aus org_dir)
      - topic_info_labels.csv (aus topics_dir)
    """

    print("\n=== [prepare_topic_colors_and_edges] ===")

    # --- Pfade aus env ---
    org_dir = Path(env["outputs"]["org_dir"])
    topics_dir = Path(env["outputs"]["topics_dir"])
    viz_dir = Path(env["outputs"].get("viz_dir", org_dir / "viz"))
    viz_dir.mkdir(parents=True, exist_ok=True)

    pies_path = org_dir / "cluster_topic_pies_summary.csv"
    flows_path = org_dir / "topic_flows_filtered.csv"
    topic_info_path = topics_dir / "topic_info_labels.csv"

    for p in [pies_path, flows_path]:
        assert p.exists(), f"❌ Datei fehlt: {p}"
    print(f"[ok] Eingabedateien gefunden.")

    # --- Daten laden ---
    pies = pd.read_csv(pies_path)
    flows = pd.read_csv(flows_path)
    labels = pd.read_csv(topic_info_path) if topic_info_path.exists() else pd.DataFrame()

    # --- Farbpalette erstellen ---
    topic_ids = sorted(
        set(pies["thread_topic_id"].dropna().unique())
        | set(flows["thread_topic_id"].dropna().unique())
    )

    cmap = cm.get_cmap("tab20", max(10, len(topic_ids)))
    topic_colors = {
        int(t): mcolors.to_hex(cmap(i % cmap.N))
        for i, t in enumerate(topic_ids)
    }

    # --- Labels anreichern (optional) ---
    if not labels.empty:
        if "Topic" in labels.columns:
            labels = labels.rename(columns={"Topic": "thread_topic_id"})
        elif "topic_id" in labels.columns:
            labels = labels.rename(columns={"topic_id": "thread_topic_id"})

    df_colors = (
        pd.DataFrame({"topic_id": list(topic_colors.keys()), "color": list(topic_colors.values())})
        .merge(labels, on="thread_topic_id", how="left") if not labels.empty
        else pd.DataFrame({"topic_id": list(topic_colors.keys()), "color": list(topic_colors.values())})
    )

    out_colors = viz_dir / "topic_colors.csv"
    df_colors.to_csv(out_colors, index=False)
    print(f"[write] {out_colors}")

    # --- Edges exportieren ---
    edges = flows.rename(
        columns={
            "sender_cluster": "source",
            "recipient_cluster": "target",
            "thread_topic_id": "topic_id",
        }
    )
    edges["color"] = edges["topic_id"].map(topic_colors)
    q_cap = edges["weight"].quantile(0.90) if len(edges) else 1.0
    q_cap = q_cap if q_cap > 0 else 1.0
    edges["width"] = 1.0 + (np.minimum(edges["weight"], q_cap) / q_cap) * 7.0
    edges["edge_id"] = (
        edges["source"].astype(str) + "||" +
        edges["target"].astype(str) + "||" +
        edges["topic_id"].astype(str)
    )

    out_edges = viz_dir / "edges_topics.csv"
    edges.to_csv(out_edges, index=False)
    print(f"[write] {out_edges}")

    print(f"✅ [prepare_topic_colors_and_edges] abgeschlossen. Topics={len(topic_colors)}")


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    prepare_topic_colors_and_edges(env)