# 📡 Complaint Radar — customer-complaint intelligence for Indian delivery apps

**🔗 Live demo: https://complaint-radar-8xsgbewhv94vvhhrxmrcwx.streamlit.app/**

An automated pipeline that collects **live Google Play reviews of Swiggy, Zomato, Blinkit and Zepto every day**,
understands them (including Hindi/Hinglish), and answers three business questions:

1. **What should each company fix first?** Complaint themes ranked by estimated rating impact, with 95% confidence ranges.
2. **Who is losing customers on what?** A like-for-like competitor benchmark.
3. **Is something breaking right now?** Statistical spike alerts on sudden jumps in a complaint theme.

## Findings (Nov 2026, ~9,000 reviews with real complaint text across 4 apps)
- **Customer support and refund problems are a top-3 rating drag for every app**, not just delivery speed.
- **Swiggy:** customer support (~0.28★ potential gain) and late delivery (~0.28★) are statistically tied; refunds follow (~0.21★).
- **Zepto:** support is the clear #1 (~0.35★), then refunds (~0.25★) and wrong/missing items (~0.17★).
- Confidence ranges matter: where two bars' ranges overlap, the dashboard tells you not to rank them.

*(Numbers change daily as new reviews arrive. Check the live app for current values.)*

## What makes this more than a dashboard
| Piece | What it does | Where |
|---|---|---|
| Automated data | GitHub Actions runs every day, merges new reviews into a growing history (no manual downloads) | `.github/workflows/daily.yml`, `radar/collect.py` |
| Multilingual NLP | Sentence embeddings (`paraphrase-multilingual-MiniLM`) understand Hinglish like "refund nahi mila" | `radar/features.py` |
| Measured models | Evaluated against baselines on held-out / hand-labelled data (below) | `radar/models.py`, `reports/evaluation.json` |
| Honest uncertainty | Bootstrap 95% confidence intervals on every impact number | `radar/analyze.py` |
| Early warning | Two-proportion z-test: last 3 days vs. previous 3 weeks, per app and theme | `radar/analyze.py` |

## Model results (measured, not assumed)
**Sentiment** (4,793 held-out reviews, labels from star ratings):
| Model | Accuracy | Macro F1 |
|---|---|---|
| Multilingual embedding classifier (ours) | **0.932** | **0.917** |
| VADER (English rule-based baseline) | 0.852 | 0.823 |

**Complaint-theme detection** (180 hand-labelled reviews, 5-fold cross-validated):
| Method | Micro F1 | Macro F1 |
|---|---|---|
| A. Keyword rules | 0.688 | 0.646 |
| B. Zero-shot embeddings | 0.464 | 0.398 |
| C. Supervised: embeddings + rules (used) | **0.701** | **0.667** |

**Honest reading:** the embedding model is a clear win for *sentiment*. For *themes*, C beats the keyword rules by only about 0.01 F1,
which is within noise on 180 examples, and zero-shot did poorly. Rules are still better on a few themes (e.g. food quality).
The multilingual model's main benefit is handling Hinglish; a bigger labelled set is the next step.

## Power BI + SQL layer
The same analysis is also exported as a **star schema** (dimension, fact and bridge tables) for BI tools and SQL:
- `powerbi/data/*.csv`, `powerbi/measures.dax`, `powerbi/ComplaintRadar-Dark.json`, `powerbi/BUILD_GUIDE.md`: a Power BI data model with DAX measures and a dark theme.
- `sql/queries.sql`: nine analytical queries (CTEs, window functions, ranking, rolling averages, spike detection). `sql/run_queries.py` runs them and
  **cross-checks the SQL impact numbers against the Python pipeline (all 40 app × theme rows match)**.
- Rebuild the exports with `python -m radar.export_powerbi`.

## Pipeline
```
radar.collect  ->  data/reviews.csv        (Google Play, 4 apps, merged + de-duplicated, last 90 days)
radar.models   ->  models/*.joblib, reports/evaluation.json   (train + evaluate; run when labels change)
radar.analyze  ->  data/analyzed.csv, impact.csv, alerts.csv  (score every review, impact + CIs, spikes)
app.py         ->  Streamlit dashboard (7 tabs)
```

## Run locally
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-pipeline.txt
python -m radar.collect --backfill      # first run: ~5,000 reviews per app
python -m radar.models                  # train + evaluate (~5 min on a laptop)
python -m radar.analyze
streamlit run app.py
```
Add a company by adding one line to `APPS` in `radar/config.py`.

## Limitations (stated up front)
- **Gold labels were assigned by an AI assistant**, not independent human annotators, and the set is small (180) and complaint-heavy. Spot-check `data/gold_labels.csv` before relying on the theme scores.
- Sentiment labels come from star ratings, which are noisy (a 1★ review can praise the product).
- About 55% of collected reviews are 1–2 word reviews ("good app") and are excluded from analysis.
- The impact score shows association, not proven cause. Play Store reviews skew negative, so absolute ratings understate satisfaction.
- Rare themes (<5 labelled examples, e.g. offers) fall back to keyword rules.
- The history window is short (Blinkit ~5 days at first backfill), so early cross-app comparisons use only the shared dates; the daily job widens it over time.

## Tech
Python · pandas · scikit-learn · fastembed (ONNX) · Streamlit · Plotly · GitHub Actions
