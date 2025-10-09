# === pipe/topics_flow/pipeline_master.py ===
"""
Master-Pipeline für Topics & Cluster-Flows im Leak-Project.
Läuft durch: setup → topic_flows_between_clusters → topic_flows →
prepare_topic_colors_and_edges → export_network_viz
"""

import runpy
from pipe.topics_flow.setup_env import setup_environment

def run_stage(name, module):
    print(f"\n\n=== [STEP: {name}] ===")
    try:
        runpy.run_module(module, run_name="__main__")
        print(f"✅ [OK] {name} abgeschlossen.")
    except Exception as e:
        print(f"❌ [ERROR in {name}] {type(e).__name__}: {e}")
        raise

def main():
    print("\n=== [TOPIC-FLOW MASTER PIPELINE START] ===")
    env = setup_environment()
    print("[env ok]", env["REPO_DIR"])

    stages = [
        ("topic_flows_between_clusters", "pipe.topics_flow.topic_flows_between_clusters"),
        ("topic_flows", "pipe.topics_flow.topic_flows"),
        ("prepare_topic_colors_and_edges", "pipe.topics_flow.prepare_topic_colors_and_edges"),
        ("export_network_viz", "pipe.topics_flow.export_network_viz")
    ]

    for name, module in stages:
        run_stage(name, module)

    print("\n=== [PIPELINE COMPLETE ✅] ===")
    print("Outputs:")
    print(" - reports/orgchart/topic_flows_filtered.csv")
    print(" - reports/orgchart/viz/org_topics_hier.html")

if __name__ == "__main__":
    main()