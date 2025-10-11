import yaml, re, csv, os, json
from pathlib import Path
from collections import defaultdict

# === Pfade ===
BASE = Path(__file__).resolve().parent
PIPELINE = BASE / "pipeline.yaml"
DEFAULT = BASE.parent / "config" / "default.yaml"
LOCAL = BASE.parent / "config" / "local.yaml"
REPORT_CSV = BASE / "pipeline_audit.csv"
REPORT_MD = BASE / "pipeline_report.md"

# === Hilfsfunktionen ===
def load_yaml(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep))
        else:
            items.append(new_key)
    return items

def search_in_files(root, term):
    """Zählt, wie oft ein Begriff in allen .py-Dateien unterhalb von root vorkommt."""
    count, hits = 0, []
    for p in Path(root).rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if term in txt:
            c = txt.count(term)
            count += c
            hits.append((p.name, c))
    return count, hits

# === 1. YAMLs laden ===
pipe = load_yaml(PIPELINE)
default = load_yaml(DEFAULT)
local = load_yaml(LOCAL)

modules = pipe.get("modules", [])
print(f"[ok] {len(modules)} Module geladen aus {PIPELINE.name}")

# === 2. Config-Keys erfassen ===
cfg_keys = set(flatten_dict(default))
cfg_keys |= set(flatten_dict(local))  # lokale Keys überschreiben ggf.
cfg_keys = {k.replace("cfg.", "") for k in cfg_keys}

# === 3. Pfadreferenzen in pipeline.yaml erfassen ===
pipeline_text = PIPELINE.read_text(encoding="utf-8")
ref_pattern = re.compile(r"\b(cfg\.[\w\.\/]+|outputs\.[\w\.\/]+|paths\.[\w\.\/]+|reports\.[\w\.\/]+)")
refs = sorted(set(ref_pattern.findall(pipeline_text)))

# === 4. Abgleich Config ↔ Pipeline ===
missing = [r for r in refs if not any(r in k for k in cfg_keys)]
defined_not_used = [k for k in cfg_keys if not any(k in r for r in refs)]

# === 5. Input/Output-Mapping ===
outputs = []
for m in modules:
    mod_name = m.get("name", "")
    for out in m.get("io", {}).get("output", []) or []:
        outputs.append({
            "output_name": out.get("name"),
            "producer": mod_name,
            "desc": out.get("desc", "")
        })

# === 6. Suche in Code ===
root_code = BASE.parent / "pipe"
for o in outputs:
    term = o["output_name"]
    if not term:
        o.update({"usage_count": 0, "used_in": [], "status": "⚠️ unknown"})
        continue
    count, hits = search_in_files(root_code, term)
    o.update({
        "usage_count": count,
        "used_in": [h[0] for h in hits],
        "status": "✅ used" if count > 0 else "⚠️ unused"
    })

# === 7. Reports schreiben ===
with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["output_name", "producer", "usage_count", "used_in", "status", "desc"])
    writer.writeheader()
    writer.writerows(outputs)

used_count = sum(o["usage_count"] > 0 for o in outputs)
unused_count = len(outputs) - used_count

with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write(f"# Pipeline Audit Report\n\n")
    f.write(f"**Module:** {len(modules)}\n")
    f.write(f"**Outputs geprüft:** {len(outputs)}\n")
    f.write(f"**Verwendete Outputs:** {used_count}\n")
    f.write(f"**Unbenutzte Outputs:** {unused_count}\n\n")

    f.write("## ⚠️ Fehlende Pfade in default.yaml\n")
    if missing:
        f.writelines(f"- {m}\n" for m in missing)
    else:
        f.write("_Keine fehlenden Pfade gefunden._\n")

    f.write("\n## 🧹 Unbenutzte Pfade in default.yaml\n")
    if defined_not_used:
        f.writelines(f"- {m}\n" for m in defined_not_used)
    else:
        f.write("_Keine unbenutzten Pfade._\n")

    f.write("\n## 🧩 Unbenutzte Outputs laut Code-Scan\n")
    for o in outputs:
        if o["status"] == "⚠️ unused":
            f.write(f"- {o['output_name']} ({o['producer']}) — {o['desc']}\n")

print(f"[done] Audit abgeschlossen ✅")
print(f"→ CSV: {REPORT_CSV}")
print(f"→ MD : {REPORT_MD}")
