"""
Exports the analysis results as a clean STAR SCHEMA for Power BI (and a SQLite copy for SQL).

    python -m radar.export_powerbi

Writes  powerbi/data/*.csv   (load these into Power BI Desktop)
        powerbi/radar.db     (same tables in SQLite, used by sql/queries.sql; not committed to git)

Model (one-to-many arrows point away from the dimension tables):

    dim_app ──< fact_reviews >── dim_date
                     │
                     └──< bridge_review_theme >── dim_theme        (a review can mention several themes)
    dim_app ──< fact_impact >── dim_theme                          (pre-computed rating gain + 95% bootstrap CI)
    dim_app ──< fact_alerts >── dim_theme                          (spike-detection results)
    model_results                                                  (measured accuracy of the NLP models)
"""
import json
import sqlite3

import pandas as pd

from radar.config import DATA, REPORTS, ROOT, THEME_NAMES

OUT = ROOT / "powerbi" / "data"
GROUPS = {"Late / slow delivery": "Operations", "Order cancelled by app": "Operations", "Wrong / missing items": "Operations",
          "Delivery partner conduct": "Operations", "Refund / money issues": "Money", "High fees / pricing": "Money",
          "Offers / membership": "Money", "Customer support": "Service", "Food / product quality": "Product",
          "App bugs / login / crash": "Technology"}
APP_COLORS = {"Swiggy": "#FC8019", "Zomato": "#E23744", "Blinkit": "#F8CB46", "Zepto": "#8025FB"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(DATA / "analyzed.csv", parse_dates=["date"])
    impact = pd.read_csv(DATA / "impact.csv", parse_dates=["first_date", "last_date"])
    alerts = pd.read_csv(DATA / "alerts.csv")

    # ---- dimensions ----
    dim_app = pd.DataFrame({"app_key": range(1, len(APP_COLORS) + 1), "app_name": list(APP_COLORS),
                            "brand_color": list(APP_COLORS.values())})
    dim_app["category"] = dim_app["app_name"].map({"Swiggy": "Food + Instamart", "Zomato": "Food delivery",
                                                   "Blinkit": "Quick commerce", "Zepto": "Quick commerce"})
    app_key = dict(zip(dim_app["app_name"], dim_app["app_key"]))

    dim_theme = pd.DataFrame({"theme_key": range(1, len(THEME_NAMES) + 1), "theme_name": THEME_NAMES})
    dim_theme["theme_group"] = dim_theme["theme_name"].map(GROUPS)
    theme_key = dict(zip(dim_theme["theme_name"], dim_theme["theme_key"]))

    days = pd.date_range(a["date"].min().normalize(), a["date"].max().normalize(), freq="D")
    dim_date = pd.DataFrame({"date": days})
    dim_date["date_key"] = dim_date["date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["year"] = dim_date["date"].dt.year
    dim_date["month_number"] = dim_date["date"].dt.month
    dim_date["month_name"] = dim_date["date"].dt.strftime("%b")
    dim_date["day"] = dim_date["date"].dt.day
    dim_date["weekday"] = dim_date["date"].dt.day_name()
    dim_date["weekday_number"] = dim_date["date"].dt.day_of_week + 1
    dim_date["is_weekend"] = dim_date["weekday_number"] >= 6
    dim_date = dim_date[["date_key", "date", "year", "month_number", "month_name", "day", "weekday", "weekday_number", "is_weekend"]]

    # ---- fact_reviews ----
    theme_cols = ["theme::" + t for t in THEME_NAMES]
    f = pd.DataFrame({
        "review_key": a["app"] + "|" + a["review_id"].astype(str),
        "app_key": a["app"].map(app_key),
        "date_key": a["date"].dt.strftime("%Y%m%d").astype(int),
        "review_time": a["date"],
        "rating": a["rating"].astype(int),
        "thumbs_up": a["thumbs_up"].astype(int),
        "sentiment_score": a["sentiment"],
        "sentiment_label": a["sentiment_label"],
        "word_count": a["text"].astype(str).str.split().str.len(),
        "theme_count": a[theme_cols].sum(axis=1).astype(int),
        "review_text": a["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 400),
    })

    # ---- bridge (long format: one row per review x theme mention) ----
    long = a[["app", "review_id"] + theme_cols].melt(id_vars=["app", "review_id"], var_name="theme", value_name="hit")
    long = long[long["hit"]]
    bridge = pd.DataFrame({"review_key": long["app"] + "|" + long["review_id"].astype(str),
                           "theme_key": long["theme"].str.replace("theme::", "", regex=False).map(theme_key)})

    # ---- precomputed facts ----
    fi = impact.assign(app_key=impact["app"].map(app_key), theme_key=impact["theme"].map(theme_key))
    fact_impact = fi[["app_key", "theme_key", "reviews", "share_pct", "avg_rating", "pct_1_star", "gain", "gain_lo", "gain_hi",
                      "n_app_reviews", "first_date", "last_date"]]
    fa = alerts.assign(app_key=alerts["app"].map(app_key), theme_key=alerts["theme"].map(theme_key))
    fact_alerts = fa[["app_key", "theme_key", "recent_share_pct", "baseline_share_pct", "change_pp", "z",
                      "recent_reviews", "baseline_reviews", "is_spike"]]

    # ---- model quality ----
    ev = json.loads((REPORTS / "evaluation.json").read_text())
    rows = [("Sentiment", "Multilingual embedding classifier", "Accuracy", ev["sentiment"]["embedding_model_accuracy"]),
            ("Sentiment", "Multilingual embedding classifier", "Macro F1", ev["sentiment"]["embedding_model_f1"]),
            ("Sentiment", "VADER baseline", "Accuracy", ev["sentiment"]["vader_accuracy"]),
            ("Sentiment", "VADER baseline", "Macro F1", ev["sentiment"]["vader_f1"])]
    for key, label in [("A_keyword_rules", "Keyword rules"), ("B_zero_shot", "Zero-shot embeddings"),
                       ("C_supervised_embeddings", "Supervised (embeddings + rules)")]:
        rows += [("Themes", label, "Micro F1", ev["themes"][key]["micro_f1"]), ("Themes", label, "Macro F1", ev["themes"][key]["macro_f1"])]
    model_results = pd.DataFrame(rows, columns=["task", "model", "metric", "value"])

    tables = {"dim_app": dim_app, "dim_theme": dim_theme, "dim_date": dim_date, "fact_reviews": f,
              "bridge_review_theme": bridge, "fact_impact": fact_impact, "fact_alerts": fact_alerts, "model_results": model_results}
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")        # BOM so Power BI reads Hindi/emoji correctly
        print(f"{name:22s} {len(df):>7,} rows")

    db = ROOT / "powerbi" / "radar.db"
    db.unlink(missing_ok=True)
    with sqlite3.connect(db) as con:
        for name, df in tables.items():
            d = df.copy()
            for c in d.columns:
                if pd.api.types.is_datetime64_any_dtype(d[c]):
                    d[c] = d[c].dt.strftime("%Y-%m-%d %H:%M:%S")
            d.to_sql(name, con, index=False)
    print("SQLite:", db)


if __name__ == "__main__":
    main()
