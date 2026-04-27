import bz2
import json
import re
import gc
from pathlib import Path
from tqdm import tqdm
import mwxml
import mwparserfromhell

# add this at the top of your script
import os, psutil
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)   # Windows only
print("CPU priority set to HIGH")
# ── config ──────────────────────────────────────────────
DUMP_FILE  = "C:/Users/bhowm/Downloads/enwiki-latest-pages-articles.xml.bz2"
OUTPUT_DIR = Path("chemistry_data")
OUTPUT_DIR.mkdir(exist_ok=True)
BATCH_SIZE  = 5000       # flush RAM every 5000 articles
JSON_INDENT = 2          # full readable JSON
MAX_TEXT    = None       # full article text — no cuts
    # limit text per article slightly

CHEMISTRY_TEMPLATES = {
    "chembox", "drugbox", "infobox drug",
    "infobox chemical", "infobox chemical compound",
    "infobox element", "infobox mineral", "infobox polymer",
    "infobox protein", "infobox enzyme",
    "infobox pharmaceutical", "infobox vitamin",
    "infobox pesticide",
}

# ── helpers ──────────────────────────────────────────────
def is_chemistry(text: str) -> bool:
    t = text.lower()
    return any(tmpl in t for tmpl in CHEMISTRY_TEMPLATES)

def extract_infobox(wikicode) -> dict:
    """Extract ALL infobox fields — no filtering."""
    props = {}
    for template in wikicode.filter_templates():
        tname = template.name.strip().lower()
        if not any(t in tname for t in CHEMISTRY_TEMPLATES):
            continue
        for param in template.params:
            key = str(param.name).strip()   # keep original key names
            try:
                val = mwparserfromhell.parse(
                    str(param.value)
                ).strip_code().strip()
                if val:
                    props[key] = val        # save ALL fields
            except:
                continue
    return props

def extract_plain_text(wikicode) -> str:
    """Full article text — no length cap."""
    text = wikicode.strip_code()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()                     # NO [:MAX_TEXT] limit

def extract_categories(text: str) -> list:
    """All categories — no filtering."""
    return re.findall(r'\[\[Category:([^\]]+)\]\]', text, re.IGNORECASE)

# ── disk space guard (safety only) ───────────────────────
def check_disk_space(min_gb=8.0):
    import shutil
    _, _, free = shutil.disk_usage(OUTPUT_DIR)
    free_gb = free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(
            f"\n⚠ Only {free_gb:.1f} GB free! "
            f"Need at least {min_gb} GB free to continue safely.\n"
            f"Free up space and restart."
        )
    return free_gb

# ── save batch ───────────────────────────────────────────
def save_batch(articles: list, texts: list, count: int):
    if not articles:
        return

    # full indented JSON — larger files, human readable
    json_file = OUTPUT_DIR / f"chemistry_batch_{count}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=JSON_INDENT, ensure_ascii=False)

    # full plain text corpus
    txt_file = OUTPUT_DIR / f"chemistry_text_{count}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.writelines(texts)

# ── main loop ────────────────────────────────────────────
def extract_chemistry(dump_file: str):
    print(f"Opening : {dump_file}")
    print(f"Output  : {OUTPUT_DIR.resolve()}\n")

    # check we have enough space for 3–5 GB output
    free = check_disk_space(min_gb=8.0)
    print(f"Disk free: {free:.1f} GB — OK to proceed\n")

    dump     = mwxml.Dump.from_file(bz2.open(dump_file, "rb"))
    articles = []
    texts    = []
    found    = 0
    skipped  = 0
    errors   = 0

    with tqdm(desc="Scanning", unit=" pages", dynamic_ncols=True) as pbar:
        for page in dump.pages:
            pbar.update(1)
            pbar.set_postfix(found=found, skipped=skipped)

            if page.namespace != 0:
                continue

            for revision in page:
                raw_text = revision.text or ""

                if not is_chemistry(raw_text):
                    skipped += 1
                    continue

                try:
                    wikicode   = mwparserfromhell.parse(raw_text)
                    infobox    = extract_infobox(wikicode)

                    if not infobox:
                        skipped += 1
                        continue

                    plain_text = extract_plain_text(wikicode)   # full text
                    categories = extract_categories(raw_text)    # all categories

                    record = {
                        "title":      page.title,
                        "categories": categories,
                        "infobox":    infobox,
                        "text":       plain_text,               # NO limit
                    }

                    articles.append(record)
                    texts.append(
                        f"# {page.title}\n\n{plain_text}\n\n---\n"
                    )
                    found += 1

                    # save every 10,000 articles
                    if found % BATCH_SIZE == 0:
                        save_batch(articles, texts, found)
                        articles.clear()
                        texts.clear()
                        gc.collect()                # free RAM after batch

                        free = check_disk_space(min_gb=5.0)
                        print(f"\n✓ Saved {found} articles | "
                              f"Disk free: {free:.1f} GB")

                except Exception as e:
                    errors += 1
                    continue

    # save final remaining batch
    save_batch(articles, texts, found)

    print(f"""
╔══════════════════════════════════╗
  Extraction Complete!
  Found   : {found:,} chemistry articles
  Skipped : {skipped:,} non-chemistry pages
  Errors  : {errors:,}
  Output  : {OUTPUT_DIR.resolve()}
╚══════════════════════════════════╝
""")

if __name__ == "__main__":
    extract_chemistry(DUMP_FILE)