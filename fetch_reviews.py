"""
STEP 1 - DATA COLLECTION
Downloads the latest real user reviews of an Android app from Google Play.

Run:  python fetch_reviews.py                      (default app: Swiggy)
      python fetch_reviews.py com.application.zomato 4000
"""
import sys
import pandas as pd
from google_play_scraper import reviews, Sort

APP_ID = sys.argv[1] if len(sys.argv) > 1 else "in.swiggy.android"
COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 3000


def fetch(app_id: str, count: int) -> pd.DataFrame:
    # Google Play returns reviews in pages; this helper handles the paging for us.
    result, _ = reviews(app_id, lang="en", country="in", sort=Sort.NEWEST, count=count)
    df = pd.DataFrame(result)
    # keep only the columns we need and give them friendlier names
    df = df[["reviewId", "content", "score", "thumbsUpCount", "at"]]
    df.columns = ["review_id", "text", "rating", "thumbs_up", "date"]
    df["app_id"] = app_id
    return df


if __name__ == "__main__":
    print(f"Fetching up to {COUNT} reviews for {APP_ID} ...")
    data = fetch(APP_ID, COUNT)
    out = f"data/reviews_{APP_ID.replace('.', '_')}.csv"
    data.to_csv(out, index=False)
    print(f"Saved {len(data)} reviews to {out}")
    print(f"Date range: {data['date'].min()} -> {data['date'].max()}")
