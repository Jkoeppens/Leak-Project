# ============================================================
# Leak-Project Configuration Loader – stabile Merge-Version
# ============================================================

from pathlib import Path
import yaml, os

# --- 🔧 robustes rekursives Merge ---
def deep_update(base, updates):
    """Rekursives Deep-Merge mit sauberem Rückschreiben."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


# --- 🔧 Platzhalter-Resolver ---
def resolve_placeholders(cfg):
    """Ersetzt {root}-Platzhalter rekursiv in allen relevanten Sektionen."""
    root = cfg["paths"]["root"]
    for section in ["paths", "outputs", "reports", "ingest"]:
        if section in cfg:
            for k, v in cfg[section].items():
                if isinstance(v, str) and "{root}" in v:
                    cfg[section][k] = v.replace("{root}", root)
    return cfg


# --- 🚀 Hauptfunktion ---
def load_config(
    default_path="config/default.yaml",
    local_path="/content/drive/MyDrive/leak-project/config/local.yaml",
    root_override=None
):
    default_path = Path(default_path)
    local_path = Path(local_path)

    # 1️⃣ Default laden
    cfg = yaml.safe_load(open(default_path))

    # 2️⃣ Lokale Anpassungen laden (falls vorhanden)
    if local_path.exists():
        local_cfg = yaml.safe_load(open(local_path))
        cfg = deep_update(cfg, local_cfg)

    # 3️⃣ Root festlegen
    root = (
        root_override
        or os.environ.get("LEAK_ROOT")
        or cfg["paths"].get("root")
        or str(default_path.parent.parent.resolve())
    )
    cfg["paths"]["root"] = root

    # 4️⃣ Platzhalter ersetzen
    cfg = resolve_placeholders(cfg)
    return cfg