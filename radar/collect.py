"""
STEP 1 - AUTOMATED COLLECTION
Pulls the newest reviews of every app in config.APPS and MERGES them into one growing history file
(data/reviews.csv). Run it daily and the dataset keeps growing by itself.

    python -m radar.collect            # normal run (newest 1500 per app)
    python -m radar.collect --backfill # first run: go deeper (5000 per app)
"""
import argparse
import sys
import time

import pandas as pd
from google_play_scraper import Sort, reviews

from radar.config import APPS, DATA

HISTORY = DATA / "reviews.csv"
COLS = ["review_id", "app", "text", "rating", "thumbs_up", "date"]


def fetch(app_name: str, app_id: str, count: int, retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            rows, _ = reviews(app_id, lang="en", country="in", sort=Sort.NEWEST, count=count)
            df = pd.DataFrame(rows)[["reviewId", "content", "score", "thumbsUpCount", "at"]]
            df.columns = ["review_id", "text", "rating", "thumbs_up", "date"]
            df["app"] = app_name
            return df[COLS]
        except Exception as e:                      # network hiccups happen; retry politely
            print(f"  {app_name}: attempt {attempt} failed ({e})", file=sys.stderr)
            time.sleep(5 * attempt)
    return pd.DataFrame(columns=COLS)


def run(count: int) -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    old = pd.read_csv(HISTORY, parse_dates=["date"]) if HISTORY.exists() else pd.DataFrame(columns=COLS)
    new_parts = []
    for name, app_id in APPS.items():
        print(f"Fetching {name} ...")
        part = fetch(name, app_id, count)
        print(f"  got {len(part)}")
        new_parts.append(part)
    merged = pd.concat([old] + new_parts, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    before = len(old)
    merged = (merged.drop_duplicates(subset=["app", "review_id"], keep="last")
                    .sort_values(["app", "date"]).reset_index(drop=True))
    cutoff = merged["date"].max() - pd.Timedelta(days=90)           # keep the repo small: last 90 days only
    merged = merged[merged["date"] >= cutoff]
    merged.to_csv(HISTORY, index=False)
    print(f"History: {before} -> {len(merged)} reviews (+{len(merged) - before} new)")
    print(merged.groupby("app")["date"].agg(["count", "min", "max"]).to_string())
    return merged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="fetch more history (first run)")
    args = ap.parse_args()
    run(5000 if args.backfill else 1500)
