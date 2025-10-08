"""
pipe.setup_env
--------------
Einheitliches Environment-Setup für Leak-Project:
- erkennt Colab-Umgebung
- klont oder aktualisiert das Repo (optional)
- kopiert lokale Config (z. B. von Drive)
- lädt YAML-Config
- gibt aufgelöste Pfade und Basisinfos zurück
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
except ImportError:
    raise ImportError("Bitte installiere PyYAML (pip install pyyaml)")

# ---------------------------------------------------------------------
# Colab Detection
# ---------------------------------------------------------------------
def in_colab() -> bool:
    """Erkennt, ob Code in einer Colab-Umgebung läuft."""
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------
# Git-Handling
# ---------------------------------------------------------------------
def ensure_repo(repo_url: str, repo_dir: Path) -> Path:
    """Klonen oder aktualisieren des Git-Repos."""
    if not repo_dir.exists():
        print(f"[git] clone -> {repo_dir}")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)
    else:
        print(f"[git] repo exists -> {repo_dir}")
        try:
            subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)
        except Exception as e:
            print("[git] pull skipped:", e)
    return repo_dir


# ---------------------------------------------------------------------
# Config-Handling
# ---------------------------------------------------------------------
def copy_local_config(src_config: Path, dst_config: Path) -> None:
    """Kopiert lokale YAML-Config, falls sie sich verändert hat."""
    dst_config.parent.mkdir(parents=True, exist_ok=True)
    if not src_config.exists():
        raise FileNotFoundError(f"Config nicht gefunden: {src_config}")
    if (not dst_config.exists()) or (src_config.read_bytes() != dst_config.read_bytes()):
        shutil.copyfile(src_config, dst_config)
        print(f"[config] kopiert -> {dst_config}")
    else:
        print(f"[config] bereits aktuell -> {dst_config}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Lädt YAML-Config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(p: str, root: Path) -> Path:
    """Hilfsfunktion: relative Pfade relativ zu root auflösen."""
    pth = Path(p)
    return pth if pth.is_absolute() else (root / pth)


# ---------------------------------------------------------------------
# Main-Bootstrap
# ---------------------------------------------------------------------
def setup_environment(
    repo_url: str = "https://github.com/Jkoeppens/Leak-Project.git",
    config_src: Path | None = None,
) -> Dict[str, Any]:
    """
    Führt das komplette Setup durch und gibt ein Summary-Dict zurück.
    """
    colab = in_colab()
    repo_dir = Path("/content/Leak-Project") if colab else Path.cwd()

    ensure_repo(repo_url, repo_dir)

    # sys.path aktualisieren
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    # Config-Quelle bestimmen
    if config_src is None:
        config_src = (
            Path("/content/drive/MyDrive/leak-project/config/local.yaml")
            if colab
            else Path.home() / "leak-project/config/local.yaml"
        )

    dst_config = repo_dir / "config" / "local.yaml"
    copy_local_config(config_src, dst_config)

    cfg = load_config(dst_config)

    # Pfade aus cfg auflösen
    paths_cfg = cfg.get("paths", {})
    root = Path(paths_cfg.get("root", str(config_src.parent.parent))).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"paths.root existiert nicht: {root}")

    clean_dir = resolve_path(paths_cfg.get("clean_dir", "data_clean"), root)
    levels_csv = resolve_path(paths_cfg.get("levels", "config/levels.csv"), root)

    outputs_cfg = cfg.get("outputs", {})
    topics_dir = resolve_path(outputs_cfg.get("topics_dir", "reports/topics"), root)
    org_dir = resolve_path(outputs_cfg.get("org_dir", "reports/org"), root)
    signif_dir = resolve_path(outputs_cfg.get("signif_dir", "reports/signif"), root)

    # Existenz prüfen / anlegen
    required_files = {
        "events.csv": clean_dir / "events.csv",
        "event_actor.csv": clean_dir / "event_actor.csv",
        "levels.csv": levels_csv,
    }
    print("\n[check files]")
    for name, p in required_files.items():
        print(f"  - {name}: {p} {'✅' if p.exists() else '❌'}")

    for d in [topics_dir, org_dir, signif_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summary = {
        "IN_COLAB": colab,
        "REPO_DIR": str(repo_dir),
        "DST_CONFIG": str(dst_config),
        "paths": {
            "root": str(root),
            "clean_dir": str(clean_dir),
            "levels_csv": str(levels_csv),
        },
        "outputs": {
            "topics_dir": str(topics_dir),
            "org_dir": str(org_dir),
            "signif_dir": str(signif_dir),
        },
    }
    print("\n[summary]\n" + json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    setup_environment()
