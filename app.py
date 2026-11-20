"""
Complaint Radar v2 - multi-app customer-complaint intelligence.
Run:  streamlit run app.py
Reads the tables produced by the pipeline (radar.collect -> radar.models -> radar.analyze).
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
st.set_page_config(page_title="Complaint Radar", page_icon="📡", layout="wide")


@st.cache_data
def load():
    df = pd.read_csv(ROOT / "data/analyzed.csv", parse_dates=["date"])
    impact = pd.read_csv(ROOT / "data/impact.csv", parse_dates=["first_date", "last_date"])
    alerts = pd.read_csv(ROOT / "data/alerts.csv")
    ev = json.loads((ROOT / "reports/evaluation.json").read_text()) if (ROOT / "reports/evaluation.json").exists() else {}
    return df, impact, alerts, ev


df, impact, alerts, ev = load()
theme_cols = [c for c in df.columns if c.startswith("theme::")]
themes = [c[7:] for c in theme_cols]
apps = sorted(df["app"].unique())

with st.sidebar:
    st.title("📡 Complaint Radar")
    st.caption("Live customer-complaint intelligence for Indian quick-commerce & food-delivery apps. "
               "Data refreshes automatically every day from Google Play.")
    app = st.selectbox("Focus app", apps, index=apps.index("Swiggy") if "Swiggy" in apps else 0)
    st.divider()
    st.metric("Reviews analysed", f"{len(df):,}")
    st.caption(f"Latest review: {df['date'].max():%d %b %Y, %H:%M}")
    n_spikes = int(alerts["is_spike"].sum()) if len(alerts) else 0
    st.metric("Active complaint spikes", n_spikes)

a = df[df["app"] == app]
imp = impact[impact["app"] == app].sort_values("gain", ascending=False)

st.title(f"{app}: what should they fix first?")
st.caption(f"{len(a):,} reviews · {a['date'].min():%d %b} → {a['date'].max():%d %b %Y}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg rating (recent reviews)", f"{a['rating'].mean():.2f} ★")
c2.metric("1-star share", f"{(a['rating'] == 1).mean():.0%}")
c3.metric("Negative sentiment", f"{(a['sentiment_label'] == 'Negative').mean():.0%}")
if len(imp):
    c4.metric("#1 problem", imp.iloc[0]["theme"], f"+{imp.iloc[0]['gain']:.2f}★ if fixed")

tabs = st.tabs(["🎯 Fix-first", "⚔️ Competitor benchmark", "🚨 Spike alerts", "📈 Trends", "💬 Voice of customer",
                "🧪 Model quality", "🔎 Search reviews"])

# ---- 1. fix-first with confidence intervals ----
with tabs[0]:
    st.subheader("Complaint themes ranked by business impact")
    st.write("**Impact** = how much the average rating would rise if unhappy customers in a theme became as satisfied as everyone "
             "else. Bars show the **95% confidence range** (bootstrap): when two ranges overlap, don't claim one is clearly worse.")
    d = imp.sort_values("gain")
    fig = go.Figure(go.Bar(x=d["gain"], y=d["theme"], orientation="h", marker_color="#d1495b",
                           error_x=dict(type="data", symmetric=False, array=d["gain_hi"] - d["gain"], arrayminus=d["gain"] - d["gain_lo"])))
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Potential rating gain (★)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(imp[["theme", "reviews", "share_pct", "avg_rating", "pct_1_star", "gain", "gain_lo", "gain_hi"]].rename(columns={
        "theme": "Theme", "reviews": "Reviews", "share_pct": "% of reviews", "avg_rating": "Avg ★", "pct_1_star": "% 1-star",
        "gain": "Gain if fixed (★)", "gain_lo": "CI low", "gain_hi": "CI high"}), hide_index=True, width="stretch")

# ---- 2. benchmark ----
with tabs[1]:
    st.subheader("Who is losing customers on what? Apps compared on the same dates")
    lo = max(impact.groupby("app")["first_date"].min()); hi = min(impact.groupby("app")["last_date"].max())
    win = df[(df["date"] >= lo) & (df["date"] <= hi)]
    st.write(f"Fair comparison window: **{lo:%d %b} → {hi:%d %b %Y}** (the dates all apps have data for) · {len(win):,} reviews.")
    share = pd.DataFrame({t: win.groupby("app")["theme::" + t].mean() * 100 for t in themes})
    heat = px.imshow(share.round(1), text_auto=True, aspect="auto", color_continuous_scale="Reds",
                     labels=dict(color="% of reviews"))
    heat.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(heat, width="stretch")
    summ = win.groupby("app").agg(reviews=("rating", "size"), avg_rating=("rating", "mean"),
                                  pct_1_star=("rating", lambda s: (s == 1).mean() * 100),
                                  negative_pct=("sentiment_label", lambda s: (s == "Negative").mean() * 100)).round(2)
    st.dataframe(summ, width="stretch")
    worst = share.idxmax(axis=1)
    st.info("**Reading the heatmap:** darker = a bigger share of that app's reviews mention the problem. "
            + " ".join(f"{ap}'s biggest complaint is **{worst[ap]}**." for ap in share.index))

# ---- 3. alerts ----
with tabs[2]:
    st.subheader("Early-warning: complaint themes that suddenly jumped")
    st.write("Last 3 days vs. the previous 3 weeks per app. A theme is flagged when the jump is statistically "
             "significant (z ≥ 2.5) **and** at least 3 percentage points. This is how an ops team would catch an outage or bad policy within a day.")
    spikes = alerts[alerts["is_spike"]]
    if len(spikes):
        st.error(f"{len(spikes)} spike(s) detected")
        st.dataframe(spikes.drop(columns=["is_spike"]), hide_index=True, width="stretch")
    else:
        st.success("No significant spikes right now. Everything is within the normal range.")
    with st.expander("All app × theme changes (including non-significant)"):
        st.dataframe(alerts.drop(columns=["is_spike"]), hide_index=True, width="stretch")

# ---- 4. trends ----
with tabs[3]:
    st.subheader("Daily trends")
    daily = df.groupby([df["date"].dt.floor("D"), "app"]).agg(avg_rating=("rating", "mean"), n=("rating", "size")).reset_index()
    daily = daily[daily["n"] >= 20]
    st.plotly_chart(px.line(daily, x="date", y="avg_rating", color="app", markers=True,
                            title="Average rating per day (days with ≥20 reviews)"), width="stretch")
    pick = st.multiselect("Theme volume over time (focus app)", themes, default=themes[:2])
    if pick:
        rows = []
        for t in pick:
            s = a.groupby(a["date"].dt.floor("D"))["theme::" + t].mean().mul(100).rename("share").reset_index().assign(theme=t)
            rows.append(s)
        st.plotly_chart(px.line(pd.concat(rows), x="date", y="share", color="theme", labels={"share": "% of that day's reviews"}),
                        width="stretch")

# ---- 5. voice of customer ----
with tabs[4]:
    t = st.selectbox("Theme", themes)
    sub = a[a["theme::" + t]].sort_values("thumbs_up", ascending=False)
    st.write(f"**{len(sub)} {app} reviews** mention this · avg {sub['rating'].mean():.2f}★. Most-upvoted first (upvotes = others agree).")
    for _, r in sub.head(8).iterrows():
        st.markdown(f"> {'★' * int(r['rating'])} **{r['thumbs_up']} 👍**: {r['text']}")

# ---- 6. model quality ----
with tabs[5]:
    st.subheader("How good is the NLP? Measured, not assumed.")
    if not ev:
        st.warning("No evaluation report found.")
    else:
        s = ev.get("sentiment", {})
        st.markdown("#### Sentiment: multilingual embedding model vs. VADER (English rule-based baseline)")
        st.caption(f"Held-out test set of {s.get('n_test', 0):,} reviews, labels from star ratings (1-2★ negative, 4-5★ positive).")
        st.dataframe(pd.DataFrame({"Model": ["Embedding classifier (ours)", "VADER baseline"],
                                   "Accuracy": [s.get("embedding_model_accuracy"), s.get("vader_accuracy")],
                                   "Macro F1": [s.get("embedding_model_f1"), s.get("vader_f1")]}), hide_index=True)
        th = ev.get("themes")
        if th:
            st.markdown("#### Theme detection: three methods on a hand-labelled gold set")
            st.caption(f"{th['n_gold']} reviews labelled by hand with their complaint themes; supervised results are 5-fold cross-validated "
                       "(no model is scored on reviews it trained on). Labels were assigned by an AI assistant and should be spot-checked.")
            st.dataframe(pd.DataFrame({"Method": ["A. Keyword rules", "B. Zero-shot embeddings", "C. Supervised (embeddings + rules)"],
                                       "Micro F1": [th["A_keyword_rules"]["micro_f1"], th["B_zero_shot"]["micro_f1"], th["C_supervised_embeddings"]["micro_f1"]],
                                       "Macro F1": [th["A_keyword_rules"]["macro_f1"], th["B_zero_shot"]["macro_f1"], th["C_supervised_embeddings"]["macro_f1"]]}),
                         hide_index=True)
            pt = pd.DataFrame({"Keyword rules": th["per_theme_f1_A"], "Supervised": th["per_theme_f1_C"],
                               "Labelled examples": th["positives_per_theme"]})
            st.markdown("**Per-theme F1** (themes with very few labelled examples are noisy):")
            st.dataframe(pt, width="stretch")
        st.markdown("**Limitations:** star-rating labels are noisy (a 1★ review can praise the product); the gold set is small and "
                    "complaint-heavy; the impact score shows association, not proof of cause; Google Play reviews skew negative.")

# ---- 7. search ----
with tabs[6]:
    st.subheader("Search the reviews")
    q = st.text_input("Describe a problem, e.g. 'refund not received after cancelled order'")
    if q:
        @st.cache_resource
        def index(texts, key):
            v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=2)
            return v, v.fit_transform(texts)
        scope = a if st.checkbox(f"Only {app}", value=True) else df
        v, m = index(tuple(scope["text"].astype(str)), (app, len(scope), q == ""))
        sims = cosine_similarity(v.transform([q]), m).ravel()
        hits = scope.assign(match=sims).sort_values("match", ascending=False).head(6)
        hits = hits[hits["match"] > 0]
        for _, r in hits.iterrows():
            st.markdown(f"> **{r['app']}** {'★' * int(r['rating'])}: {r['text']}  \n*match {r['match']:.2f}*")
        if hits.empty:
            st.write("No close matches.")
