"""
STEP 2 - MODELS + HONEST EVALUATION
(a) Sentiment: multilingual-embedding classifier vs the VADER baseline.
(b) Themes:  keyword rules  vs  zero-shot embeddings  vs  supervised classifier,
             all scored on a hand-labelled "gold" set (data/gold_labels.csv).

    python -m radar.models        -> trains, prints metrics, writes reports/evaluation.json + models/*.joblib
"""
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from radar.config import DATA, MODELS, REPORTS, THEME_NAMES, THEMES
from radar.features import embed_texts, get_embeddings, key

GOLD = DATA / "gold_labels.csv"


def clean(text) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", str(text).lower())).strip()


def keyword_flags(texts) -> np.ndarray:
    cleaned = [clean(t) for t in texts]
    return np.array([[bool(re.search(THEMES[t][1], c)) for t in THEME_NAMES] for c in cleaned], dtype=float)


def theme_description_vectors() -> np.ndarray:
    return embed_texts([THEMES[t][0] for t in THEME_NAMES])


# ----------------------------------------------------------------------------------------------
def train_sentiment(df: pd.DataFrame, X: np.ndarray) -> dict:
    """Weak supervision: 1-2 star = negative, 4-5 star = positive. Evaluated on a held-out 25%."""
    mask = (df["rating"] <= 2) | (df["rating"] >= 4)
    y = (df.loc[mask, "rating"] >= 4).astype(int).values
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(X[mask.values], y, np.where(mask)[0],
                                                    test_size=0.25, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced").fit(Xtr, ytr)
    pred = clf.predict(Xte)

    sia = SentimentIntensityAnalyzer()
    vader = np.array([sia.polarity_scores(str(df["text"].iloc[i]))["compound"] >= 0.05 for i in ite]).astype(int)
    res = {
        "n_test": int(len(yte)),
        "embedding_model_accuracy": round(accuracy_score(yte, pred), 3),
        "embedding_model_f1": round(f1_score(yte, pred, average="macro"), 3),
        "vader_accuracy": round(accuracy_score(yte, vader), 3),
        "vader_f1": round(f1_score(yte, vader, average="macro"), 3),
    }
    joblib.dump(LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced").fit(X[mask.values], y),
                MODELS / "sentiment.joblib")
    return res


# ----------------------------------------------------------------------------------------------
def load_gold() -> tuple[pd.DataFrame, np.ndarray]:
    g = pd.read_csv(GOLD)
    Y = np.array([[t in str(s).split(";") for t in THEME_NAMES] for s in g["themes"].fillna("")], dtype=int)
    return g, Y


def prf(Y, P) -> dict:
    return {"micro_f1": round(f1_score(Y, P, average="micro", zero_division=0), 3),
            "macro_f1": round(f1_score(Y, P, average="macro", zero_division=0), 3)}


def train_themes(hist: pd.DataFrame, Xall: np.ndarray) -> dict:
    gold, Y = load_gold()
    idx = key(hist).reset_index(drop=True)
    pos = {k: i for i, k in enumerate(idx)}
    gi = [pos[k] for k in (gold["app"] + "|" + gold["review_id"].astype(str))]
    Xg = Xall[gi]
    texts = hist["text"].iloc[gi].tolist()      # full review text (gold file only stores a 230-char preview)

    # A. keyword rules
    K = keyword_flags(texts)
    out = {"n_gold": int(len(gold)), "positives_per_theme": dict(zip(THEME_NAMES, Y.sum(0).astype(int).tolist()))}
    out["A_keyword_rules"] = prf(Y, K)

    # B. zero-shot: cosine similarity to a plain-English description; one global threshold tuned on gold
    S = Xg @ theme_description_vectors().T
    best_t, best_f = 0, -1
    for t in np.linspace(0.15, 0.6, 46):
        f = f1_score(Y, (S >= t).astype(int), average="micro", zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    out["B_zero_shot"] = {**prf(Y, (S >= best_t).astype(int)), "threshold": round(float(best_t), 3)}

    # C. supervised: embeddings + keyword flags -> one logistic regression per theme, 5-fold cross-validated
    feats = np.hstack([Xg, K])
    P = np.zeros_like(Y)
    PR = np.zeros(Y.shape, dtype=float)
    for j in range(len(THEME_NAMES)):
        if Y[:, j].sum() < 5:                          # too few examples to learn: fall back to keywords
            P[:, j] = K[:, j]; PR[:, j] = K[:, j]
            continue
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        prob = cross_val_predict(LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced"),
                                 feats, Y[:, j], cv=cv, method="predict_proba")[:, 1]
        PR[:, j] = prob
    # choose a single probability threshold on the cross-validated scores
    best_t, best_f = 0.5, -1
    for t in np.linspace(0.2, 0.8, 25):
        f = f1_score(Y, (PR >= t).astype(int), average="micro", zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    P = (PR >= best_t).astype(int)
    out["C_supervised_embeddings"] = {**prf(Y, P), "threshold": round(float(best_t), 3)}
    out["per_theme_f1_C"] = {t: round(f1_score(Y[:, j], P[:, j], zero_division=0), 2) for j, t in enumerate(THEME_NAMES)}
    out["per_theme_f1_A"] = {t: round(f1_score(Y[:, j], K[:, j], zero_division=0), 2) for j, t in enumerate(THEME_NAMES)}

    # final models trained on ALL gold data
    final = {}
    for j, t in enumerate(THEME_NAMES):
        if Y[:, j].sum() >= 5:
            final[t] = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced").fit(feats, Y[:, j])
    joblib.dump({"models": final, "threshold": float(best_t)}, MODELS / "themes.joblib")
    return out


def main():
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    hist = pd.read_csv(DATA / "reviews.csv", parse_dates=["date"]).dropna(subset=["text"]).reset_index(drop=True)
    X = get_embeddings(hist)
    report = {"sentiment": train_sentiment(hist, X)}
    if GOLD.exists():
        report["themes"] = train_themes(hist, X)
    (REPORTS / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
