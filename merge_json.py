import json
import glob
from pathlib import Path

OUTPUT_DIR = Path("chemistry_data")

print("=== Merging JSON files only (TXT not needed) ===\n")

all_articles = []

# load all JSON batches
for f in sorted(glob.glob(str(OUTPUT_DIR / "chemistry_batch_*.json"))):
    with open(f, encoding="utf-8") as fp:
        batch = json.load(fp)
        all_articles.extend(batch)
        size_mb = Path(f).stat().st_size / (1024**2)
        print(f"✓ {Path(f).name} → {len(batch):,} articles ({size_mb:.1f} MB)")

print(f"\nTotal articles: {len(all_articles):,}")

# build JSONL training file
jsonl_path = OUTPUT_DIR / "chemistry_train.jsonl"
count = 0

with open(jsonl_path, "w", encoding="utf-8") as out:
    for article in all_articles:
        title   = article.get("title", "")
        text    = article.get("text", "")
        infobox = article.get("infobox", {})

        # ── Entry 1: full rich text ──────────────────────
        rich = f"### {title}\n\n"
        if infobox:
            rich += "Chemical Properties:\n"
            for k, v in infobox.items():
                rich += f"  {k}: {v}\n"
            rich += "\n"
        if text:
            rich += f"{text}\n"

        out.write(json.dumps({"text": rich}, ensure_ascii=False) + "\n")
        count += 1

        # ── Entry 2: Q&A from infobox ────────────────────
        qa_map = {
            ("formula", "molecularformula"):
                f"What is the molecular formula of {title}?",
            ("molarmass", "molarweight"):
                f"What is the molar mass of {title}?",
            ("smiles",):
                f"What is the SMILES notation of {title}?",
            ("casno", "casnumber"):
                f"What is the CAS number of {title}?",
            ("boilingpt",):
                f"What is the boiling point of {title}?",
            ("meltingpt",):
                f"What is the melting point of {title}?",
            ("density",):
                f"What is the density of {title}?",
            ("appearance",):
                f"What does {title} look like?",
        }

        for keys, question in qa_map.items():
            for key in keys:
                val = infobox.get(key)
                if val:
                    out.write(json.dumps({
                        "text": f"Q: {question}\nA: {val}"
                    }, ensure_ascii=False) + "\n")
                    count += 1
                    break

size_mb = jsonl_path.stat().st_size / (1024**2)
print(f"""
╔══════════════════════════════════════╗
  Done!
  Articles merged  : {len(all_articles):,}
  Training entries : {count:,}
  Output file      : chemistry_train.jsonl
  Size             : {size_mb:.1f} MB
╚══════════════════════════════════════╝
""")
print("Now upload chemistry_train.jsonl to Google Drive!")