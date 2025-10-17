# ============================================================
# Leak-Project Configuration Loader
# Compatible with both local and Colab/Drive environments
# Branch: consolidate/ingest-core
# ============================================================

from pathlib import Path
import yaml
from string import Template
import os

# ------------------------------------------------------------
# Helper: deep merge (local.yaml overrides default.yaml)
# ------------------------------------------------------------
def deep_update(base, updates):
    """Recursively update dict 'base' with values from 'updates'."""
    for k, v in updates.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            deep_update(base[k], v)
        else:
            base[k] = v

# ------------------------------------------------------------
# Helper: resolve {root} placeholders in paths
# ------------------------------------------------------------
def resolve_placeholders(cfg):
    """Replace {root} placeholders in all path-like config sections."""
    root = cfg["paths"]["root"]
    for section in ["paths", "outputs", "reports", "ingest"]:
        if section in cfg:
            for k, v in cfg[section].items():
                if isinstance(v, str):
                    cfg[section][k] = Template(v).safe_substitute(root=root)
    return cfg

# ------------------------------------------------------------
# Main loader function
# ------------------------------------------------------------
def load_config(
    default_path: str = "config/default.yaml",
    local_path: str = "config/local.yaml",
    root_override: str = None
):
    """
    Load Leak-Project configuration.
    1. Loads default.yaml (always required)
    2. Optionally merges local.yaml (if present)
    3. Replaces {root} placeholders using:
       priority = root_override > local.yaml > default.yaml > repo root
    """

    default_path = Path(default_path)
    local_path = Path(local_path)
    cfg = yaml.safe_load(open(default_path))

    # Optional: merge local.yaml
    if local_path.exists():
        local_cfg = yaml.safe_load(open(local_path))
        deep_update(cfg, local_cfg)

    # Determine root path
    root = (
        root_override
        or os.environ.get("LEAK_ROOT")
        or cfg["paths"].get("root")
        or str(default_path.parent.parent.resolve())
    )
    cfg["paths"]["root"] = root

    # Replace placeholders
    cfg = resolve_placeholders(cfg)

    return cfg

# ------------------------------------------------------------
# CLI / Debug entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    print(f"[config] Using root: {cfg['paths']['root']}")
    print(f"[config] Raw dir: {cfg['paths'].get('raw_dir')}")
    print(f"[config] Clean dir: {cfg['paths'].get('clean_dir')}")
