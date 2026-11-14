"""List the interface strings of the web app, or (with a second argument) fail if any lacks a Bulgarian entry.

    python web/scripts/check_i18n.py web          # list
    python web/scripts/check_i18n.py web check    # exit 1 when something is missing
"""
import glob
import json
import re
import sys

root = sys.argv[1]
pat = re.compile(r'\bt\(\s*"((?:[^"\\]|\\.)*)"')
seen = []
for f in sorted(glob.glob(root + "/components/*.tsx")):
    for m in pat.finditer(open(f, encoding="utf-8").read()):
        if m.group(1) not in seen:
            seen.append(m.group(1))
extra = ["Tomorrow", "Battery", "Track record", "Market", "Model", "Bulgaria", "Romania", "Greece", "Hungary",
         "LightGBM, monthly refit (previous model)", "LightGBM, weekly refit", "LightGBM on asinh prices",
         "LightGBM on asinh prices, weekly refit", "LEAR (per-hour lasso)", "Adaptive ensemble of three",
         "Adaptive ensemble of four (in production)", "Ridge, weekly refit", "Level-adjusted LightGBM, weekly refit"]
for x in extra:
    if x not in seen:
        seen.append(x)
if len(sys.argv) > 2:      # check mode: report strings missing from the Bulgarian dictionary
    bg = open(root + "/lib/i18n-bg.ts", encoding="utf-8").read()
    missing = [s for s in seen if json.dumps(s, ensure_ascii=False) + ":" not in bg]
    print(len(seen), "strings,", len(missing), "missing")
    for s in missing:
        print("  MISSING:", s)
    sys.exit(1 if missing else 0)
else:
    print(len(seen))
    for s in seen:
        print(s)
