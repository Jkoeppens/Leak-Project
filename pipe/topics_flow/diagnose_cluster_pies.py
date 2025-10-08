"""
pipe.diagnose_cluster_pies
==========================
Explorative Analyse und Plausibilitätsprüfung der Cluster–Topic-Verteilungen.

Erwartet:
  reports/orgchart/cluster_topic_pies.csv
  reports/orgchart/cluster_topic_pies_summary.csv  (optional)

Zeigt:
  - Übersicht und Häufigkeitsstatistik
  - Dominanz / Monokultur-Cluster
  - Globale Topic-Gewichtung
  - Heatmap & Histogramme
  - Heuristik-Warnungen
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def diagnose_cluster_pies(env: dict) -> None:
    org_dir = Path(env["outputs"]["org_dir"])
    OUT_PIES         = org_dir / "cluster_topic_pies.csv"
    OUT_PIES_SUMMARY = org_dir / "cluster_topic_pies_summary.csv"

    assert OUT_PIES.exists(), f"Fehlt: {OUT_PIES}"
    pies = pd.read_csv(OUT_PIES)
    summary = pd.read_csv(OUT_PIES_SUMMARY) if OUT_PIES_SUMMARY.exists() else None

    # --- Grundcheck ---
    req = {"cluster_id","thread_topic_id","weight","share"}
    missing = req - set(pies.columns)
    assert not missing, f"Spalten fehlen in pies: {missing}"

    print("=== Überblick ===")
    n_clusters = pies["cluster_id"].nunique()
    n_topics   = pies["thread_topic_id"].nunique()
    print(f"Cluster: {n_clusters} | Topics (in Verteilungen): {n_topics} | Zeilen: {len(pies)}")

    # --- Gesamtgewicht je Cluster ---
    tot = pies.groupby("cluster_id")["weight"].sum().rename("cluster_weight").reset_index()
    print("\n=== Top 10 Cluster nach Gesamtgewicht (Textmenge) ===")
    print(tot.sort_values("cluster_weight", ascending=False).head(10).to_string(index=False))

    # --- Dominanz: höchster Topic-Anteil je Cluster ---
    dom = (pies.sort_values(["cluster_id","share"], ascending=[True, False])
               .groupby("cluster_id").head(1)
               .rename(columns={"thread_topic_id":"dominant_topic","share":"dominant_share"})
               [["cluster_id","dominant_topic","dominant_share","weight"]])
    print("\n=== Dominanter Topic pro Cluster (Top 10 nach Anteil) ===")
    print(dom.sort_values("dominant_share", ascending=False).head(10).to_string(index=False))

    mono = dom[dom["dominant_share"] >= 0.80]
    print(f"\nMonokultur-Cluster (share ≥ 0.80): {len(mono)}")
    if len(mono):
        print(mono.sort_values("dominant_share", ascending=False).head(20).to_string(index=False))

    # --- Topic-Abdeckung global ---
    topic_tot = pies.groupby("thread_topic_id")["weight"].sum().rename("topic_weight").reset_index()
    topic_tot["topic_weight_share_global"] = topic_tot["topic_weight"] / topic_tot["topic_weight"].sum()
    print("\n=== Top 10 Topics nach globaler Textmenge ===")
    print(topic_tot.sort_values("topic_weight", ascending=False).head(10).to_string(index=False))

    # --- Heatmap: Cluster × Topic (Top-N Themen) ---
    TOP_N_TOPICS = 12
    top_topics = set(topic_tot.sort_values("topic_weight", ascending=False)
                              .head(TOP_N_TOPICS)["thread_topic_id"].tolist())
    heat = pies[pies["thread_topic_id"].isin(top_topics)].pivot_table(
        index="cluster_id", columns="thread_topic_id", values="share", aggfunc="max", fill_value=0.0
    )

    plt.figure()
    plt.imshow(heat.values, aspect="auto")
    plt.title(f"Heatmap: Anteil (share) Top-{TOP_N_TOPICS} Topics pro Cluster")
    plt.xlabel("thread_topic_id")
    plt.ylabel("cluster_id")
    plt.xticks(ticks=range(len(heat.columns)), labels=heat.columns, rotation=90)
    plt.yticks([])
    plt.colorbar(label="share")
    plt.show()

    # --- Verteilung Cluster-Gewichte ---
    plt.figure()
    plt.hist(tot["cluster_weight"], bins=40)
    plt.title("Verteilung Gesamtgewicht (Textmenge) pro Cluster")
    plt.xlabel("Textmenge (weight)")
    plt.ylabel("Anzahl Cluster")
    plt.show()

    # --- Cross-Check Summary ---
    if summary is not None:
        print("\n=== Summary (Top-N je Cluster) – Beispiel ===")
        cols_show = [c for c in ["cluster_id","thread_topic_id","weight","share","Name","topic_label"]
                     if c in summary.columns]
        print(summary.sort_values(["cluster_id","weight"], ascending=[True, False])
                     .head(20)[cols_show].to_string(index=False))

    # --- Heuristik-Checks ---
    issues = []
    zero_clusters = tot[tot["cluster_weight"] <= 0]["cluster_id"].tolist()
    if zero_clusters:
        issues.append(f"{len(zero_clusters)} Cluster mit weight=0")

    topic_cluster_counts = pies.groupby("thread_topic_id")["cluster_id"].nunique()
    rare_topics = topic_cluster_counts[topic_cluster_counts == 1].index.tolist()
    if len(rare_topics) > 0:
        issues.append(f"{len(rare_topics)} Topics kommen nur in 1 Cluster vor (Spezial/Artefakt?)")

    many_extreme = (dom["dominant_share"] > 0.95).sum()
    if many_extreme > max(5, 0.1 * n_clusters):
        issues.append(f"Viele Monokulturen (dominant_share>0.95): {many_extreme}")

    print("\n=== Heuristik-Checks ===")
    print("OK" if not issues else " | ".join(issues))


if __name__ == "__main__":
    from pipe.topics_flow.setup_env import setup_environment
    env = setup_environment()
    diagnose_cluster_pies(env)