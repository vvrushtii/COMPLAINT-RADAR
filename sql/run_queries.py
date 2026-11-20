"""Runs every query in sql/queries.sql against powerbi/radar.db, prints a preview, and cross-checks the impact query
against the Python pipeline's numbers (data/impact.csv)."""
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
con = sqlite3.connect(ROOT / "powerbi" / "radar.db")
sql = (ROOT / "sql" / "queries.sql").read_text(encoding="utf-8")
blocks = re.split(r"^-- name: ", sql, flags=re.M)[1:]
pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60)
results = {}
for b in blocks:
    head, _, body = b.partition("\n")
    name = head.split("|")[0].strip()
    df = pd.read_sql_query(body, con)
    results[name] = df
    print(f"\n=== {name}  ({len(df)} rows) ===")
    print(df.head(8).to_string(index=False))

# cross-check: SQL gain == Python pipeline gain
sql_gain = results["02_theme_impact"].rename(columns={"app_name": "app", "theme_name": "theme", "gain": "gain_sql"})
py = pd.read_csv(ROOT / "data" / "impact.csv")[["app", "theme", "gain"]]
m = py.merge(sql_gain[["app", "theme", "gain_sql"]], on=["app", "theme"], how="outer", indicator=True)
diff = (m["gain"] - m["gain_sql"]).abs().max()
print(f"\nCROSS-CHECK vs Python pipeline: {len(m)} app x theme rows, unmatched={(m['_merge'] != 'both').sum()}, max abs diff={diff:.4f}")
sys.exit(0 if (m["_merge"] == "both").all() and diff < 0.002 else 1)
