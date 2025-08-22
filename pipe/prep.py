# pipe/prep.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .io import (
    load_edges, load_levels,
    filter_internal_mode, apply_edge_filters, ego_subgraph,
    basic_report, save_report, hist_weight, hist_degree
)

def prepare_edges_and_report(cfg: dict) -> tuple[pd.DataFrame, str, dict]:
    """
    Lädt Rohkanten, wendet Internal-/Edge-/Ego-Filter an,
    speichert prepared-Edges und erzeugt einen Basis-Report + Plots.
    Returns: (E_prepared, prepared_csv_path, report_dict)
    """
    # 1) Laden
    E0 = load_edges(cfg["paths"]["edges"])

    f_int = cfg.get("filters", {}).get("internal", {})
    f_edg = cfg.get("filters", {}).get("edges", {})
    f_sub = cfg.get("filters", {}).get("subgraph", {})

    # 2) Extern-Logik
    E1 = filter_internal_mode(
        E0,
        mode=f_int.get("mode","outside_supernode"),
        domains=f_int.get("domains", []),
        outside_label=f_int.get("outside_label","Outside world"),
    )

    # 3) Weitere Filter
    E2 = apply_edge_filters(
        E1,
        min_weight=int(f_edg.get("min_weight", 1)),
        require_reciprocal=bool(f_edg.get("require_reciprocal", False)),
    )

    # 4) Optionaler Ego-Subgraph
    E3 = ego_subgraph(
        E2,
        seeds=f_sub.get("seed_emails") or [],
        hops=int(f_sub.get("hops", 0)),
    )

    # 5) Persistenz (prepared)
    prep_csv = Path(cfg["paths"]["edges_prepared"])
    prep_csv.parent.mkdir(parents=True, exist_ok=True)
    E3.to_csv(prep_csv, index=False)

    # 6) Report & Plots
    cutdir = cfg["outputs"]["cutoffs_dir"]
    H = None
    lvl = cfg["paths"].get("levels")
    if lvl and Path(lvl).exists():
        H = load_levels(lvl)
    rep = basic_report(E3, H)
    save_report(rep, cutdir)
    hist_weight(E3, str(Path(cutdir)/"hist_weights_prepared.png"))
    hist_degree(E3, str(Path(cutdir)/"hist_degree_prepared.png"))

    return E3, str(prep_csv), rep
