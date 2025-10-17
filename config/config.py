# ============================================================
# Leak-Project Configuration Loader (fix für {root})
# ============================================================

from pathlib import Path
import yaml, os

def deep_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and k in base:
            deep_update(base[k], v)
        else:
            base[k] = v

def resolve_placeholders(cfg):
    """Ersetzt {root} Platzhalter rekursiv in allen relevanten Sektionen."""
    root = cfg["paths"]["root"]
    for section in ["paths", "outputs", "reports", "ingest"]:
        if section in cfg:
            for k, v in cfg[section].items():
                if isinstance(v, str) and "{root}" in v:
                    cfg[section][k] = v.replace("{root}", root)
    return cfg

def load_config(
    default_path="config/default.yaml",
    local_path="config/local.yaml",
    root_override=None
):
    default_path = Path(default_path)
    local_path = Path(local_path)

    cfg = yaml.safe_load(open(default_path))
    if local_path.exists():
        local_cfg = yaml.safe_load(open(local_path))
        deep_update(cfg, local_cfg)

    root = (
        root_override
        or os.environ.get("LEAK_ROOT")
        or cfg["paths"].get("root")
        or str(default_path.parent.parent.resolve())
    )
    cfg["paths"]["root"] = root
    cfg = resolve_placeholders(cfg)
    return cfg

if __name__ == "__main__":
    cfg = load_config()
    print("[config] root:", cfg["paths"]["root"])
    print("[config] raw_dir:", cfg["paths"]["raw_dir"])
    print("[config] clean_dir:", cfg["paths"]["clean_dir"])