#!/usr/bin/env python3
import os, re, sys, json, glob, time, subprocess
import pandas as pd

CANON_UNITS = r"(?:kg|g|l|ml|tsp|tbsp|cup|oz|lb|pcs|clove|can|tin|pkt|bunch)"
PAT_NUM_TO_UNIT = rf'(?<![A-Za-z])(\\d+(?:[.,]\\d+)?(?:\\s+\\d/\\d)?|\\d/\\d)\\s*({CANON_UNITS})\\b'
PAT_UNIT_TO_WORD = rf'\\b({CANON_UNITS})(?=[A-Za-z])'

CFG_FILES = [
    os.path.join("config","extracted_prices2.csv"),
    os.path.join("config","ingredient-dataset_nutrition.xlsx"),
]

def die(msg, code=1):
    print(f"\n❌ {msg}")
    sys.exit(code)

def ok(msg): print(f"✅ {msg}")
def warn(msg): print(f"⚠️  {msg}")

def exists(path): return os.path.exists(path)

def run(cmd):
    print(f"\n>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def newest(pattern_or_list, after_ts=None):
    pats = pattern_or_list if isinstance(pattern_or_list, (list,tuple)) else [pattern_or_list]
    files = []
    for p in pats:
        files.extend([f for f in glob.glob(p) if os.path.isfile(f)])
    if after_ts: files = [f for f in files if os.path.getmtime(f) >= after_ts]
    return max(files, key=os.path.getmtime) if files else None

def tokenize(line):
    if not isinstance(line,str): return []
    return [t.strip() for t in line.split("|") if t.strip()]

def metrics_for(df):
    tokens = []
    for s in df["ingredients_per_person"].fillna(""):
        tokens += tokenize(s)
    total = len(tokens)
    if total == 0:
        return {"total_tokens":0, "digit_or_unit_pct":0, "glued":0, "parens":0}
    digit_or_unit = sum(bool(re.search(r"\d", t) or re.search(rf"\b{CANON_UNITS}\b", t, flags=re.I)) for t in tokens)
    glued = sum(bool(re.search(rf"\b{CANON_UNITS}(?=[A-Za-z])", t, flags=re.I) or
                     re.search(rf"\b\d+(?:[.,]\d+)?(?:\s+\d/\d)?\s*{CANON_UNITS}(?=[A-Za-z])", t, flags=re.I)) for t in tokens)
    parens = sum(("(" in t or ")" in t) for t in tokens)
    return {
        "total_tokens": total,
        "digit_or_unit_pct": round(digit_or_unit/total*100, 2),
        "glued": glued,
        "parens": parens
    }

def strip_parens(s:str)->str:
    if not isinstance(s,str): return ""
    prev=None
    while prev!=s:
        prev=s
        s=re.sub(r"\([^)]*\)", "", s)
    return s

def fix_line(s:str)->str:
    if not isinstance(s,str): return ""
    parts = tokenize(s)
    out=[]
    for t in parts:
        t = strip_parens(t)                                     # kill parentheses
        t = re.sub(PAT_UNIT_TO_WORD, r"\\1 ", t, flags=re.I)    # unit→word spacing
        t = re.sub(PAT_NUM_TO_UNIT, r"\\1 \\2", t, flags=re.I)  # num→unit spacing
        t = re.sub(r"\s+", " ", t).strip()
        out.append(t)
    return " | ".join(out)

def fix_normalized_csv(in_csv, out_csv):
    print(f"\n🛠  Fixing glued units & parentheses in: {in_csv}")
    df = pd.read_csv(in_csv, encoding="utf-8")
    if "ingredients_per_person" not in df.columns:
        die(f"'{in_csv}' has no 'ingredients_per_person' column. Run serving.py first.")
    before = metrics_for(df)

    df["ingredients_per_person"] = df["ingredients_per_person"].apply(fix_line)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    after = metrics_for(df)
    print("\n🔎 METRICS (before → after)")
    print(f"  Total tokens:           {before['total_tokens']} → {after['total_tokens']}")
    print(f"  % with digit or unit:   {before['digit_or_unit_pct']}% → {after['digit_or_unit_pct']}%")
    print(f"  Glued unit→word count:  {before['glued']} → {after['glued']}")
    print(f"  Parentheses count:      {before['parens']} → {after['parens']}")
    if after["glued"] != 0 or after["parens"] != 0:
        warn("Post-fix still shows residual glued or parentheses. Scripts will still run, but consider patching serving.py.")
    else:
        ok("Tokens look clean (glued=0, parentheses=0).")
    return out_csv

def main():
    # 0) Basic checks
    for cfg in CFG_FILES:
        if not exists(cfg):
            warn(f"Missing config file: {cfg}")

    # 1) Figure out inputs
    recipes_csv = "recipes_base.csv" if exists("recipes_base.csv") else None
    norm_csv = "recipes_base_normalized.csv" if exists("recipes_base_normalized.csv") else None

    if not norm_csv and not recipes_csv:
        die("Could not find 'recipes_base_normalized.csv' or 'recipes_base.csv' in this folder.")

    # 2) If no normalized CSV, try to create it with serving.py
    if not norm_csv and recipes_csv:
        if not exists("serving.py"):
            die("No normalized CSV and 'serving.py' not found to create it. Place serving.py here or provide recipes_base_normalized.csv.")
        print("\n🥣 Creating normalized CSV with serving.py …")
        t0 = time.time()
        run([sys.executable, "serving.py"])
        norm_csv = newest(["*normalized*.csv", "recipes_base_normalized.csv"], after_ts=t0) \
                   or "recipes_base_normalized.csv"
        if not exists(norm_csv):
            die("serving.py did not produce a normalized CSV. Check its logs and try again.")

    ok(f"Using normalized CSV: {norm_csv}")

    # 3) Fix normalized CSV (glued + parentheses) → *_fixed.csv
    fixed_csv = os.path.splitext(norm_csv)[0] + "_fixed.csv"
    fixed_csv = fix_normalized_csv(norm_csv, fixed_csv)
    ok(f"Fixed normalized CSV → {fixed_csv}")

    # 4) Nutrition stage
    nutr_in = fixed_csv if exists(fixed_csv) else norm_csv
    print("\n🥗 Running nutrition …")
    t1 = time.time()
    if not exists("nutrition_calculator.py"):
        die("nutrition_calculator.py not found.")
    run([sys.executable, "nutrition_calculator.py", nutr_in])
    nutr_out = newest(["*improved_nutrition*.csv", "recipes_improved_nutrition.csv"], after_ts=t1) \
               or "recipes_improved_nutrition.csv"
    if not exists(nutr_out):
        die("Nutrition script did not produce an improved nutrition CSV.")
    ok(f"Nutrition CSV → {nutr_out}")

    # 5) Pricing stage
    print("\n💸 Running pricing …")
    if not exists("price_calculator.py"):
        die("price_calculator.py not found.")
    t2 = time.time()
    run([sys.executable, "price_calculator.py", nutr_out])

    report = newest("pricing_coverage_report.json", after_ts=t2) or "pricing_coverage_report.json"
    if not exists(report):
        warn("pricing_coverage_report.json not found. Your pricing script should write it.")
    else:
        with open(report, "r", encoding="utf-8") as f:
            data = json.load(f)
        baseline = float(data.get("Baseline") or data.get("baseline") or 0.0)
        strict   = float(data.get("Strict")   or data.get("strict")   or 0.0)
        very     = float(data.get("VeryStrict") or data.get("very_strict") or strict)

        print("\n📄 Coverage Report")
        print(f"  Baseline:    {baseline}%")
        print(f"  Strict:      {strict}%")
        print(f"  VeryStrict:  {very}%")
        if baseline > 0 and strict == 0:
            die("Fail-fast: Baseline>0 but Strict==0. Check unit normalization, per-kg index, and denominator rules.")
        ok("Pricing stage completed.")

    print("\n🎉 Done. Your pipeline ran end-to-end.")

if __name__ == "__main__":
    main()
