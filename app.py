"""
STEP 3 - THE DASHBOARD  (run with:  streamlit run app.py)
Customer Complaint Radar: turns thousands of app reviews into a "fix this first" list.
"""
import glob
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import analyze
import fetch_reviews

st.set_page_config(page_title="Complaint Radar", page_icon="📡", layout="wide")


# ---------- data loading ----------
files = sorted(glob.glob("data/*_analyzed.csv"))
if not files:
    st.error("No analysed data found. Run: python fetch_reviews.py  then  python analyze.py data/<file>.csv")
    st.stop()

with st.sidebar:
    st.title("📡 Complaint Radar")
    choice = st.selectbox("App", files, format_func=lambda p: os.path.basename(p).replace("reviews_", "").replace("_analyzed.csv", ""))
    st.caption("Reviews are pulled live from Google Play, so results change as customers complain.")
    st.divider()
    st.subheader("Refresh with live data")
    app_id = os.path.basename(choice).replace("reviews_", "").replace("_analyzed.csv", "")
    app_id = app_id.replace("_", ".")
    n = st.slider("Reviews to fetch", 500, 5000, 3000, 500)
    if st.button("Fetch newest reviews & re-analyse"):
        with st.spinner("Downloading reviews and running analysis..."):
            raw = fetch_reviews.fetch(app_id, n)
            raw_path = f"data/reviews_{app_id.replace('.', '_')}.csv"
            raw.to_csv(raw_path, index=False)
            analyze.run(raw_path)
        st.cache_data.clear()
        st.rerun()


@st.cache_data
def load(path: str):
    df = pd.read_csv(path, parse_dates=["date"])
    pri = pd.read_csv(path.replace("_analyzed", "_priority"))
    emg = pd.read_csv(path.replace("_analyzed", "_emerging"))
    return df, pri, emg


df, priority, emerging = load(choice)
theme_cols = [c for c in df.columns if c.startswith("theme::")]

# ---------- header + KPIs ----------
st.title("What should this company fix first?")
st.caption(f"{len(df):,} real customer reviews · {df['date'].min():%d %b %Y} → {df['date'].max():%d %b %Y}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Average rating", f"{df['rating'].mean():.2f} ★")
c2.metric("1-star reviews", f"{(df['rating'] == 1).mean():.0%}")
c3.metric("Negative sentiment", f"{(df['sentiment_label'] == 'Negative').mean():.0%}")
top = priority.iloc[0]
c4.metric("#1 problem", top["theme"], f"+{top['potential_rating_gain']:.2f}★ if fixed", delta_color="normal")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Fix-first list", "📈 Trends", "💬 Voice of customer", "🔍 Emerging topics", "🤖 Ask the reviews"])

# ---------- tab 1: priority ----------
with tab1:
    st.subheader("Complaint themes ranked by business impact")
    st.write("**Impact** = how much the app's average rating would rise if unhappy customers in that theme "
             "became as satisfied as everyone else. Big and angry beats small and angry.")
    fig = px.bar(priority.sort_values("potential_rating_gain"), x="potential_rating_gain", y="theme", orientation="h",
                 color="avg_rating", color_continuous_scale="RdYlGn", range_color=(1, 3),
                 labels={"potential_rating_gain": "Potential rating gain (★)", "theme": "", "avg_rating": "Avg ★ of theme"})
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(priority.rename(columns={
        "theme": "Theme", "reviews": "Reviews", "share_pct": "% of reviews", "avg_rating": "Avg ★",
        "pct_1_star": "% 1-star", "avg_sentiment": "Sentiment", "potential_rating_gain": "Rating gain if fixed"}),
        hide_index=True, width="stretch")
    t1, t2, t3 = priority.head(3)["theme"].tolist()
    st.success(f"**Recommendation:** focus first on **{t1}**, **{t2}** and **{t3}**. Together they account for "
               f"{priority.head(3)['potential_rating_gain'].sum():.2f}★ of potential rating gain "
               f"(current average is {df['rating'].mean():.2f}★).")

# ---------- tab 2: trends ----------
with tab2:
    st.subheader("How is customer mood changing day by day?")
    daily = df.set_index("date").resample("D").agg(avg_rating=("rating", "mean"), avg_sentiment=("sentiment", "mean"),
                                                   reviews=("rating", "size")).dropna().reset_index()
    st.plotly_chart(px.line(daily, x="date", y="avg_rating", markers=True, title="Average rating per day",
                            labels={"avg_rating": "Avg ★", "date": ""}), width="stretch")
    pick = st.multiselect("Compare theme volume over time", [c[7:] for c in theme_cols],
                          default=[priority.iloc[0]["theme"], priority.iloc[1]["theme"]])
    if pick:
        trend = pd.concat([df.assign(theme=t).loc[df["theme::" + t]].set_index("date").resample("D").size().rename("mentions")
                           .reset_index().assign(theme=t) for t in pick])
        st.plotly_chart(px.line(trend, x="date", y="mentions", color="theme", title="Daily mentions"),
                        width="stretch")
        st.caption("A sudden spike in one line = something broke that day (outage, bad promo, policy change). "
                   "This is how a company would use the tool as an early-warning system.")

# ---------- tab 3: voice of customer ----------
with tab3:
    theme = st.selectbox("Pick a theme", [c[7:] for c in theme_cols], index=0)
    subset = df[df["theme::" + theme]].sort_values("thumbs_up", ascending=False)
    st.write(f"**{len(subset)} reviews** mention this theme · avg {subset['rating'].mean():.2f}★. "
             "Most-upvoted shown first (upvotes = other customers agree).")
    for _, r in subset.head(8).iterrows():
        st.markdown(f"> {'★' * int(r['rating'])} **{r['thumbs_up']} 👍** — {r['text']}")

# ---------- tab 4: emerging ----------
with tab4:
    st.subheader("Topics discovered automatically in 1-2★ reviews")
    st.write("No keywords were given here. The algorithm (TF-IDF + KMeans) grouped similar angry reviews by itself. "
             "Useful for spotting problems the theme list doesn't cover yet.")
    for _, r in emerging.iterrows():
        with st.expander(f"{r['reviews']} reviews · {r['top_words']}"):
            st.write(r["example"])

# ---------- tab 5: ask ----------
with tab5:
    st.subheader("Ask the reviews")
    st.write("Type a question or a problem in plain English. The tool finds the most similar real reviews "
             "(semantic-style search using TF-IDF cosine similarity).")
    q = st.text_input("e.g. 'refund not received after cancelled order'")
    if q:
        @st.cache_resource
        def build_index(_texts, key):
            v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            return v, v.fit_transform(_texts)
        vec, mat = build_index(df["text"].astype(str).tolist(), choice + str(len(df)))
        sims = cosine_similarity(vec.transform([q]), mat).ravel()
        hits = df.assign(match=sims).sort_values("match", ascending=False).head(6)
        hits = hits[hits["match"] > 0]
        st.write(f"Average rating of the {len(hits)} closest reviews: **{hits['rating'].mean():.2f}★**" if len(hits) else "No close matches.")
        for _, r in hits.iterrows():
            st.markdown(f"> {'★' * int(r['rating'])} — {r['text']}  \n*match {r['match']:.2f}*")

