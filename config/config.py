# ============================================================
# Leak-Project Configuration Loader
# Compatible with both local and Colab/Drive environments
# Branch: consolidate/ingest-core
# Supports {root} placeholder syntax (Variante B)
# ============================================================

from pathlib import Path
import yaml
import os

# ------------------------------------------------------------
# Helper: deep merge (local.yaml overrides default.yaml)
# ------------------------------------------------------------
def deep_update(base: dict, updates: dict):
    """Recursively update dict 'base' with values from 'updates'."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v

# ------------------------------------------------------------
# Helper: resolve {root} placeholders in paths
# ------------------------------------------------------------
def resolve_placeholders(cfg: dict):
    """Replace {root} placeholders in all path-like config sections."""
    root = cfg["paths"]["root"]
    for section in ["paths", "outputs", "reports", "ingest"]:
        if section in cfg:
            for k, v in cfg[section].items():
                if isinstance(v, str):
                    cfg[section][k] = v.replace("{root}", root)
    return cfg

# ------------------------------------------------------------
# Main loader function
# ------------------------------------------------------------
def load_config(
    default_path: str = "config/default.yaml",
    local_path: str = "config/local.yaml",
    root_override: str = None,
):
    """
    Load Leak-Project configuration.

    Steps:
      1. Load default.yaml (always required)
      2. Merge local.yaml (if present)
      3. Determine root path:
         priority = root_override > $LEAK_ROOT > local.yaml > default.yaml > repo root
      4. Replace {root} placeholders across config sections
    """
    default_path = Path(default_path)
    local_path = Path(local_path)

    # --- 1️⃣ Load default.yaml
    with open(default_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- 2️⃣ Optional: merge local.yaml
    if local_path.exists():
        with open(local_path, "r") as f:
            local_cfg = yaml.safe_load(f)
        deep_update(cfg, local_cfg)

    # --- 3️⃣ Determine root
    root = (
        root_override
        or os.environ.get("LEAK_ROOT")
        or cfg["paths"].get("root")
        or str(default_path.parent.parent.resolve())
    )
    cfg["paths"]["root"] = root

    # --- 4️⃣ Replace {root} placeholders
    cfg = resolve_placeholders(cfg)

    return cfg

# ------------------------------------------------------------
# CLI / Debug entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    print(f"[config] Using root: {cfg['paths']['root']}")
    for key in ["raw_dir", "clean_dir", "derived_dir"]:
        if key in cfg["paths"]:
            print(f"[config] {key}: {cfg['paths'][key]}")
