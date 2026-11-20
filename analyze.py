"""
STEP 2 - ANALYSIS
Turns raw reviews into:
  1. sentiment per review                      (VADER)
  2. complaint THEMES (keyword taxonomy)       -> the priority list
  3. EMERGING topics inside negative reviews   (TF-IDF + KMeans) -> finds things we didn't think of

Run:  python analyze.py data/reviews_in_swiggy_android.csv
"""
import re
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---- 1. Theme taxonomy: theme name -> keywords/regex that signal it -------------------------
# A review can match several themes (e.g. "late delivery AND rude support").
# This is editable: add a theme, re-run, and the whole dashboard updates.
THEMES = {
    "Late / slow delivery":      r"\blate\b|delay|\bslow\b|too long|waiting|waited|hour|never arrive|not deliver|not arrive",
    "Order cancelled by app":    r"cancel",
    "Refund / money issues":     r"refund|money|charged|deduct|payment|wallet|double charge|not received my",
    "Customer support":          r"support|customer care|customer service|no response|chat bot|chatbot|helpline|agent",
    "Wrong / missing items":     r"wrong|missing|missed|incorrect|not received|spilled|spill|damaged",
    "Food quality":              r"cold food|stale|quality|taste|tasteless|hygien|raw|smell|spoil|rotten",
    "High fees / pricing":       r"fee|charge|expensive|costly|price|surge|platform fee|delivery charge|overpric|handling",
    "App bugs / login / crash":  r"\bbug|crash|login|log in|otp|hang|glitch|not working|stuck|freez|update|error|loading",
    "Delivery partner conduct":  r"delivery partner|delivery boy|rider|partner (?:was|is)|rude|behav|misbehav",
    "Offers / membership":       r"coupon|offer|discount|promo|swiggy one|membership|subscription|cashback",
    "Instamart / grocery":       r"instamart|grocery|groceries|vegetable|fruit|milk",
}

EXTRA_STOP = {"app", "swiggy", "zomato", "just", "really", "like", "get", "got", "use", "using",
              "good", "bad", "very", "also", "even", "one", "please", "pls", "worst", "great", "nice", "best"}
STOP = list(ENGLISH_STOP_WORDS | EXTRA_STOP)
N_EMERGING = 6


def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def run(path: str):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["text"]).copy()
    df["clean"] = df["text"].map(clean)
    df = df[df["clean"].str.split().str.len() >= 3].reset_index(drop=True)     # drop 1-2 word reviews

    # --- sentiment: VADER gives each review a score from -1 (angry) to +1 (happy)
    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = df["text"].map(lambda t: sia.polarity_scores(str(t))["compound"])
    df["sentiment_label"] = pd.cut(df["sentiment"], [-1.01, -0.05, 0.05, 1.01],
                                   labels=["Negative", "Neutral", "Positive"]).astype(str)

    # --- themes: one True/False column per theme
    for theme, pattern in THEMES.items():
        df["theme::" + theme] = df["clean"].str.contains(pattern, regex=True)
    df["themes"] = df.apply(lambda r: "; ".join(t for t in THEMES if r["theme::" + t]) or "Other", axis=1)

    # --- priority table: "if we fixed this theme, how much would the average rating rise?"
    # Rating gain ~= share of reviews mentioning the theme x how much lower their rating is than the rest.
    rows = []
    for theme in THEMES:
        m = df["theme::" + theme]
        if m.sum() < 10:
            continue
        share = m.mean()
        gap = max(df.loc[~m, "rating"].mean() - df.loc[m, "rating"].mean(), 0)
        rows.append({
            "theme": theme,
            "reviews": int(m.sum()),
            "share_pct": round(share * 100, 1),
            "avg_rating": round(df.loc[m, "rating"].mean(), 2),
            "pct_1_star": round((df.loc[m, "rating"] == 1).mean() * 100, 1),
            "avg_sentiment": round(df.loc[m, "sentiment"].mean(), 2),
            "potential_rating_gain": round(share * gap, 3),
        })
    priority = pd.DataFrame(rows).sort_values("potential_rating_gain", ascending=False)

    # --- emerging topics: cluster ONLY the unhappy reviews (1-2 stars) to discover patterns
    neg = df[df["rating"] <= 2]
    vec = TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=4, max_df=0.4)
    X = vec.fit_transform(neg["clean"])
    km = KMeans(n_clusters=N_EMERGING, n_init=10, random_state=42).fit(X)
    terms = vec.get_feature_names_out()
    em_rows = []
    for k in range(N_EMERGING):
        top = km.cluster_centers_[k].argsort()[::-1][:5]
        members = neg[km.labels_ == k]
        em_rows.append({
            "cluster": k,
            "top_words": ", ".join(terms[i] for i in top),
            "reviews": len(members),
            "example": members.sort_values("thumbs_up", ascending=False)["text"].iloc[0][:220],
        })
    emerging = pd.DataFrame(em_rows).sort_values("reviews", ascending=False)

    base = path.replace(".csv", "")
    df.drop(columns=["clean"]).to_csv(base + "_analyzed.csv", index=False)
    priority.to_csv(base + "_priority.csv", index=False)
    emerging.to_csv(base + "_emerging.csv", index=False)

    print(f"Reviews analysed: {len(df)} | avg rating: {df['rating'].mean():.2f} | "
          f"negative sentiment: {(df['sentiment_label'] == 'Negative').mean():.0%}")
    print("\nPRIORITY LIST\n", priority.to_string(index=False))
    print("\nEMERGING TOPICS (1-2 star reviews)\n", emerging[["top_words", "reviews"]].to_string(index=False))
    return df, priority, emerging


if __name__ == "__main__":
    run(sys.argv[1])
