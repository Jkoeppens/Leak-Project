# ============================================================
# Leak-Project Configuration Loader – stabile Version (recursive merge)
# ============================================================

from pathlib import Path
import yaml, os

# ------------------------------------------------------------
# 🔁 Rekursives Merging
# ------------------------------------------------------------
def deep_update(base: dict, updates: dict) -> dict:
    """Rekursives Update: ersetzt Werte und legt neue Keys an."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base

# ------------------------------------------------------------
# 🧩 Platzhalter-Auflösung
# ------------------------------------------------------------
def resolve_placeholders(cfg: dict) -> dict:
    """Ersetzt {root} Platzhalter rekursiv in allen relevanten Sektionen."""
    root = cfg["paths"]["root"]
    for section in ["paths", "outputs", "reports", "ingest"]:
        if section in cfg:
            for k, v in cfg[section].items():
                if isinstance(v, str) and "{root}" in v:
                    cfg[section][k] = v.replace("{root}", root)
    return cfg

# ------------------------------------------------------------
# ⚙️ Konfigurationslader
# ------------------------------------------------------------
def load_config(
    default_path="config/default.yaml",
    local_path="/content/drive/MyDrive/leak-project/config/local.yaml",
    root_override=None
):
    default_path = Path(default_path)
    local_path = Path(local_path)

    cfg = yaml.safe_load(open(default_path))
    if local_path.exists():
        local_cfg = yaml.safe_load(open(local_path))
        cfg = deep_update(cfg, local_cfg)

    # Root festlegen (local.yaml > Env > Override > Repo)
    root = (
        root_override
        or cfg["paths"].get("root")
        or os.environ.get("LEAK_ROOT")
        or str(default_path.parent.parent.resolve())
    )

    cfg["paths"]["root"] = root
    cfg = resolve_placeholders(cfg)
    return cfg