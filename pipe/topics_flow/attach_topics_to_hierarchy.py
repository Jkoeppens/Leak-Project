# === pipe/topics_flow/attach_topics_to_hierarchy.py ===
import pandas as pd
from pathlib import Path


def attach_topics_to_hierarchy(env):
    """
    Verknüpft Topics mit Hierarchie-Clustern auf Basis der Infomap-Levelstruktur.
    Erwartet in env Pfade zu:
      - org_dir (mit cluster_topic_pies_summary.csv)
      - levels_csv (Infomap-Level-Datei)
    """

    print("\n=== [attach_topics_to_hierarchy] Starte Verarbeitung ===")

    # ---- Pfade aus Environment laden ----
    org_dir = Path(env["outputs"]["org_dir"])
    topics_dir = Path(env["outputs"]["topics_dir"])

    pies_summary_path = org_dir / "cluster_topic_pies_summary.csv"
    assert pies_summary_path.exists(), f"❌ Datei fehlt: {pies_summary_path}"

    # --- Korrigierte Pfadlogik für neues setup_env ---
    levels_csv_path = (
        Path(env.get("inputs", {}).get("levels_csv"))
        if env.get("inputs", {}).get("levels_csv")
        else Path(env["paths"]["levels_csv"])
    )

    assert levels_csv_path.exists(), f"❌ Datei fehlt: {levels_csv_path}"

    # ---- Daten laden ----
    print(f"[load] pies_summary: {pies_summary_path}")
    pies_summary = pd.read_csv(pies_summary_path)

    print(f"[load] infomap_levels: {levels_csv_path}")
    H = pd.read_csv(levels_csv_path)

    # ---- Grundlegende Integritätsprüfung ----
    required_cols = ["cluster_id", "thread_topic_id", "weight"]
    missing = [c for c in required_cols if c not in pies_summary.columns]
    assert not missing, f"Spalten fehlen in pies_summary: {missing}"

    print(f"[ok] Pies: {len(pies_summary):,} Zeilen | {pies_summary['cluster_id'].nunique()} Cluster")

    # ---- Levels prüfen ----
    level_cols = [c for c in H.columns if c.startswith("level_")]
    print(f"[info] Levels erkannt: {level_cols}")

    # ---- Topics nach Hierarchie mappen ----
    # cluster_id (aus pies_summary) liegt als "1:10:143:..." vor
    pies_summary["module_path"] = pies_summary["cluster_id"].astype(str)

    # Extrahiere Hierarchieebenen
    pies_summary[["level_1", "level_2", "level_3", "level_4", "level_5"]] = (
        pies_summary["module_path"].str.split(":", expand=True).iloc[:, :5]
    )

    pies_summary["level_depth"] = pies_summary.apply(
        lambda r: sum(pd.notna([r.get(c) for c in ["level_1", "level_2", "level_3", "level_4", "level_5"]])),
        axis=1,
    )

    print(f"[ok] Hierarchieebenen extrahiert (max depth={pies_summary['level_depth'].max()})")

    # ---- Aggregation der Topics pro Ebene ----
    agg_by_level = []
    for col in [c for c in level_cols if c in pies_summary.columns]:
        tmp = (
            pies_summary.groupby([col, "thread_topic_id"], dropna=True)
            .agg(weight_sum=("weight", "sum"), n=("weight", "count"))
            .reset_index()
            .assign(level=col)
        )
        agg_by_level.append(tmp)
        print(f"[agg] Ebene {col}: {len(tmp)} Zeilen")

    agg_all = pd.concat(agg_by_level, ignore_index=True)
    print(f"[done] Gesamt: {len(agg_all):,} Zeilen über {len(level_cols)} Ebenen")

    # ---- Export ----
    out_csv = org_dir / "topic_distribution_by_level.csv"
    agg_all.to_csv(out_csv, index=False)
    print(f"[save] {out_csv}")

    print("\n✅ [attach_topics_to_hierarchy] abgeschlossen.")


# === Script Entry Point ===
if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment

    env = setup_environment()
    attach_topics_to_hierarchy(env)