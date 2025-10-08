"""
pipe.prepare_topic_colors_and_edges
===================================
Erzeugt konsistente Farben für alle Topics und bereitet zusätzliche
'Extra-Edges' für visuelle Layer (z.B. Themenflüsse über Hierarchie)
vor.

Inputs:
  - reports/orgchart/viz/topic_colors.csv (optional, wird ergänzt)
  - dataframes T (Topics / cluster-topic-pies) und F (Topic-Flows)

Outputs:
  - aktualisierte topic_colors.csv
  - Liste 'extra_edges' mit Kanten-Attributen für Visualisierung
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def prepare_topic_colors_and_edges(env: dict, T: pd.DataFrame, F: pd.DataFrame):
    viz_dir = Path(env["outputs"]["viz_dir"])
    viz_dir.mkdir(parents=True, exist_ok=True)
    tc_csv = viz_dir / "topic_colors.csv"

    # -------- Hilfsfunktion --------
    def lv_of_cluster(cid: str) -> str:
        """Level bestimmen anhand der Doppelpunkte in der Cluster-ID."""
        if not isinstance(cid, str):
            return "L1"
        return f"L{cid.count(':') + 1}"

    # -------- Farben vorbereiten --------
    topics_needed = sorted(set(T["thread_topic_id"].dropna().unique()) |
                          set(F["thread_topic_id"].dropna().unique()))

    COLOR_MAP = {}
    if tc_csv.exists():
        tc = pd.read_csv(tc_csv)
        topic_col = "topic_id" if "topic_id" in tc.columns else tc.columns[0]
        color_col = "color" if "color" in tc.columns else "color_hex"
        for _, r in tc.iterrows():
            try:
                COLOR_MAP[float(r[topic_col])] = str(r[color_col])
            except Exception:
                COLOR_MAP[str(r[topic_col])] = str(r[color_col])

    # Fehlende Topics ergänzen
    missing = [t for t in topics_needed if t not in COLOR_MAP and str(t) not in COLOR_MAP]
    if missing:
        palette = cm.get_cmap("tab20", max(10, len(missing)))
        for i, t in enumerate(missing):
            COLOR_MAP[t] = mcolors.to_hex(palette(i))

    # -------- Farbtabelle aktualisieren --------
    out_colors = pd.DataFrame({
        "topic_id": topics_needed,
        "color": [COLOR_MAP.get(t, COLOR_MAP.get(str(t), "#888888")) for t in topics_needed]
    })
    out_colors.to_csv(tc_csv, index=False)
    print(f"[write] {tc_csv} — {len(out_colors)} Topics mit Farben")

    # -------- Extra-Edges vorbereiten --------
    SMOOTH_OPTS = {"enabled": True, "type": "dynamic", "roundness": 0.2}
    DASHED = False

    extra_edges = []
    for _, r in F.iterrows():
        t = r["thread_topic_id"]
        col = COLOR_MAP.get(t, COLOR_MAP.get(str(t), "#888888"))
        extra_edges.append({
            "level": lv_of_cluster(r["sender_cluster"]),
            "src": r["sender_cluster"],
            "dst": r["recipient_cluster"],
            "width": float(r.get("width_scaled", 1.0)),
            "color": str(col),
            "label": f"{int(t) if pd.notna(t) else t} • {int(r['weight'])}",
            "arrows": "to",
            "dashes": DASHED,
            "smooth": SMOOTH_OPTS,
        })

    print(f"[ok] Extra-Edges vorbereitet: {len(extra_edges)} | Topics: {len(topics_needed)}")
    return COLOR_MAP, extra_edges


if __name__ == "__main__":
    from pipe.setup_env import setup_environment
    env = setup_environment()

    # Beispielhafter Load (du kannst in deinem Pipeline-Skript echte Frames übergeben)
    T = pd.read_csv(env["outputs"]["org_dir"] / "cluster_topic_pies.csv")
    F = pd.read_csv(env["outputs"]["org_dir"] / "topic_flows.csv")

    prepare_topic_colors_and_edges(env, T, F)