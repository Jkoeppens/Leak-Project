# === pipe/topics_flow/attach_topics_to_hierarchy.py ===
"""
Verknüpft Topics mit Hierarchie-Clustern auf Basis der Infomap-Levelstruktur.
Unterstützt zeitliche Aggregation (periodisch oder rolling).

Input:
  - cluster_topic_pies_summary.csv (oder cluster_topic_pies_summary_YYYY-MM.csv)
  - infomap_levels.csv (aus data_derived)

Output:
  - topic_distribution_by_level.csv (oder topic_distribution_by_level_YYYY-MM.csv)
"""

import pandas as pd
from pathlib import Path


def attach_topics_to_hierarchy(env):
    print("\n=== [attach_topics_to_hierarchy] Starte Verarbeitung ===")

    org_dir = Path(env["outputs"]["org_dir"])
    topics_dir = Path(env["outputs"]["topics_dir"])
    levels_csv_path = Path(env["paths"]["levels_csv"])

    # ---- Pfadprüfung ----
    assert levels_csv_path.exists(), f"❌ Datei fehlt: {levels_csv_path}"
    time_mode = env["runtime"].get("time_mode", "off")

    # ---- Hilfsfunktion: finde Pies-Dateien ----
    def find_pies_files():
        if time_mode == "off":
            return [org_dir / "cluster_topic_pies_summary.csv"]
        else:
            # Alle periodischen Dateien laden
            return sorted(org_dir.glob("cluster_topic_pies_summary_*.csv"))

    pie_files = find_pies_files()
    assert len(pie_files) > 0, f"❌ Keine cluster_topic_pies_summary-Dateien gefunden in {org_dir}"

    # ---- Lade Hierarchie ----
    levels = pd.read_csv(levels_csv_path)
    level_cols = [c for c in levels.columns if c.startswith("level_")]
    print(f"[ok] Hierarchieebenen erkannt: {level_cols}")

    # ---- Hilfsfunktion: Normalisierung der Cluster-IDs ----
    def normalize_cluster(val):
        if pd.isna(val):
            return None
        return str(val).strip()

    levels["module_path"] = levels[level_cols].astype(str).agg(":".join, axis=1)

    # ---- Iteration über Pies-Dateien ----
    for file in pie_files:
        print(f"[load] {file.name}")
        pies = pd.read_csv(file, low_memory=False)
        pies["cluster_id"] = pies["cluster_id"].astype(str).apply(normalize_cluster)

        # ---- Merging mit Hierarchie ----
        merged = pies.merge(
            levels,
            left_on="cluster_id",
            right_on="module_path",
            how="left",
        )

        # ---- Sanity: Fehlende Zuordnungen ----
        missing = merged["level_1"].isna().sum()
        if missing > 0:
            print(f"[warn] {missing} Cluster ohne Hierarchie-Zuordnung")

        # ---- Aggregation über Ebenen ----
        dist_frames = []
        for col in level_cols:
            dist = (
                merged.groupby([col, "thread_topic_id"])
                .agg(
                    weight=("weight", "sum"),
                    n_events=("n_events", "sum"),
                    share=("share", "mean"),
                )
                .reset_index()
            )
            dist["level"] = col
            dist = dist.rename(columns={col: "cluster_id"})
            dist_frames.append(dist)

        topic_dist = pd.concat(dist_frames, ignore_index=True)
        print(f"[ok] {len(topic_dist)} Zeilen aggregiert über {len(level_cols)} Ebenen")

        # ---- Zeitinformation aus Dateinamen extrahieren ----
        if time_mode != "off" and "_" in file.stem:
            period = file.stem.split("_")[-1]
        else:
            period = "ALL"

        # ---- Export ----
        out_name = (
            f"topic_distribution_by_level_{period}.csv"
            if time_mode != "off"
            else "topic_distribution_by_level.csv"
        )
        out_path = org_dir / out_name
        topic_dist.to_csv(out_path, index=False)
        print(f"[save] {out_path}")

    print("✅ [attach_topics_to_hierarchy] abgeschlossen.")


# -----------------------------------------------------
if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    attach_topics_to_hierarchy(env)