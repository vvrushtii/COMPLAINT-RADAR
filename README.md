# 📡 Complaint Radar — "What should this company fix first?"

An end-to-end data project that turns thousands of **real, live Google Play reviews** of a food-delivery app
into a ranked, business-ready fix list.

**Key finding (Swiggy, 1,164 reviews, Nov 2026):** average rating is 2.40★. Customer support, refund problems
and late delivery are the top three drivers of low ratings. Fixing them could lift the average by roughly
0.7★ (see dashboard for exact numbers; they change as new reviews arrive).

## The business problem
App teams get thousands of reviews and no time to read them. Which complaint, if solved, would improve the
rating the most? Counting complaints isn't enough: a rare but furious issue and a common mild one need to be
compared fairly.

## How it works (pipeline)
| Step | File | What happens | Skill shown |
|---|---|---|---|
| 1. Collect | `fetch_reviews.py` | Pulls newest reviews live from Google Play | Data acquisition, APIs |
| 2. Clean + sentiment | `analyze.py` | Lower-case, remove noise, VADER sentiment score per review | Text preprocessing, NLP |
| 3. Themes | `analyze.py` | 11-theme keyword taxonomy (support, refunds, delay...) | Domain modelling |
| 4. Impact score | `analyze.py` | `share of reviews × (rating of others − rating of theme)` = rating gain if fixed | Business metric design |
| 5. Discovery | `analyze.py` | TF-IDF + KMeans on 1–2★ reviews finds unknown topics | Unsupervised ML |
| 6. Dashboard | `app.py` | Streamlit app: fix-first list, trends, real quotes, search, one-click live refresh | Data storytelling, deployment |

## Run it
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python fetch_reviews.py in.swiggy.android 3000
python analyze.py data/reviews_in_swiggy_android.csv
streamlit run app.py
```
Try another app: `python fetch_reviews.py com.application.zomato 3000`, then analyse it. It appears in the dropdown.

## Limitations (say these in interviews, it shows maturity)
- Keyword themes can miss unusual wording; Hindi/Hinglish reviews are only partly captured (the KMeans step found a Hinglish cluster).
- VADER is rule-based and English-focused. A next step is a multilingual transformer model.
- The rating-gain score is an estimate from correlation, not proof of causation.
- Google Play reviews skew negative (unhappy users write more), so absolute ratings understate satisfaction.

## Next steps
Multilingual sentiment model · LLM-generated summaries per theme · scheduled daily refresh with GitHub Actions · alerting on theme spikes.
