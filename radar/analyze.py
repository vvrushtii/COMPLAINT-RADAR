"""
STEP 3 - ANALYSIS
Applies the trained models to every review and produces the tables the dashboard reads:
  data/analyzed.csv   one row per review (sentiment + theme flags)
  data/impact.csv     per app x theme: share, avg rating, rating gain if fixed, 95% bootstrap CI
  data/alerts.csv     per app x theme: complaint SPIKES in the last 3 days vs the previous weeks (z-test)

    python -m radar.analyze
"""
import joblib
import numpy as np
import pandas as pd

from radar.config import DATA, MODELS, THEME_NAMES
from radar.features import get_embeddings
from radar.models import keyword_flags

rng = np.random.default_rng(42)


def score_reviews(df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    sent = joblib.load(MODELS / "sentiment.joblib")
    df["sentiment"] = (sent.predict_proba(X)[:, 1] * 2 - 1).round(3)               # -1 angry ... +1 happy
    df["sentiment_label"] = pd.cut(df["sentiment"], [-1.01, -0.2, 0.2, 1.01], labels=["Negative", "Neutral", "Positive"]).astype(str)

    K = keyword_flags(df["text"].tolist())
    feats = np.hstack([X, K])
    th = joblib.load(MODELS / "themes.joblib")
    for j, t in enumerate(THEME_NAMES):
        if t in th["models"]:
            df["theme::" + t] = th["models"][t].predict_proba(feats)[:, 1] >= th["threshold"]
        else:                                                                       # too few labelled examples: keep the rules
            df["theme::" + t] = K[:, j].astype(bool)
    return df


def impact_table(df: pd.DataFrame, n_boot: int = 300) -> pd.DataFrame:
    """rating gain if fixed = share of reviews in the theme x (avg rating of the rest - avg rating of the theme)."""
    rows = []
    for app, g in df.groupby("app"):
        rating = g["rating"].values.astype(float)
        n = len(rating)
        for t in THEME_NAMES:
            m = g["theme::" + t].values
            if m.sum() < 10:
                continue
            def gain(idx):
                mm, rr = m[idx], rating[idx]
                if mm.sum() == 0 or (~mm).sum() == 0:
                    return 0.0
                return mm.mean() * max(rr[~mm].mean() - rr[mm].mean(), 0)
            point = gain(np.arange(n))
            boots = [gain(rng.integers(0, n, n)) for _ in range(n_boot)]
            rows.append({"app": app, "theme": t, "reviews": int(m.sum()), "share_pct": round(m.mean() * 100, 1),
                         "avg_rating": round(rating[m].mean(), 2), "pct_1_star": round((rating[m] == 1).mean() * 100, 1),
                         "gain": round(point, 3), "gain_lo": round(np.percentile(boots, 2.5), 3),
                         "gain_hi": round(np.percentile(boots, 97.5), 3), "n_app_reviews": n,
                         "first_date": g["date"].min(), "last_date": g["date"].max()})
    return pd.DataFrame(rows).sort_values(["app", "gain"], ascending=[True, False])


def spike_alerts(df: pd.DataFrame, recent_days: int = 3, baseline_days: int = 21) -> pd.DataFrame:
    """Two-proportion z-test: is the theme share in the last N days significantly above its recent baseline?"""
    rows = []
    for app, g in df.groupby("app"):
        end = g["date"].max().normalize() + pd.Timedelta(days=1)
        rec = g[g["date"] >= end - pd.Timedelta(days=recent_days)]
        base = g[(g["date"] < end - pd.Timedelta(days=recent_days)) & (g["date"] >= end - pd.Timedelta(days=recent_days + baseline_days))]
        if len(rec) < 30 or len(base) < 60:
            continue
        for t in THEME_NAMES:
            p1, p0 = rec["theme::" + t].mean(), base["theme::" + t].mean()
            p = (rec["theme::" + t].sum() + base["theme::" + t].sum()) / (len(rec) + len(base))
            se = np.sqrt(max(p * (1 - p), 1e-9) * (1 / len(rec) + 1 / len(base)))
            z = (p1 - p0) / se
            rows.append({"app": app, "theme": t, "recent_share_pct": round(p1 * 100, 1), "baseline_share_pct": round(p0 * 100, 1),
                         "change_pp": round((p1 - p0) * 100, 1), "z": round(z, 2), "recent_reviews": len(rec),
                         "baseline_reviews": len(base), "is_spike": bool(z >= 2.5 and (p1 - p0) >= 0.03)})
    return pd.DataFrame(rows).sort_values("z", ascending=False)


def main():
    hist = pd.read_csv(DATA / "reviews.csv", parse_dates=["date"]).dropna(subset=["text"]).reset_index(drop=True)
    hist = hist[hist["text"].astype(str).str.split().str.len() >= 3].reset_index(drop=True)
    X = get_embeddings(hist)
    df = score_reviews(hist, X)
    df.to_csv(DATA / "analyzed.csv", index=False)
    impact = impact_table(df)
    impact.to_csv(DATA / "impact.csv", index=False)
    alerts = spike_alerts(df)
    alerts.to_csv(DATA / "alerts.csv", index=False)
    print(f"{len(df)} reviews analysed across {df['app'].nunique()} apps")
    print(impact.groupby("app").head(3)[["app", "theme", "share_pct", "avg_rating", "gain", "gain_lo", "gain_hi"]].to_string(index=False))
    print("\nSpikes:\n", alerts[alerts["is_spike"]].to_string(index=False) if len(alerts) else "none")


if __name__ == "__main__":
    main()
