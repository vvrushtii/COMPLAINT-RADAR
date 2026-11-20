-- Complaint Radar: SQL analysis on the star schema (SQLite syntax; works in Postgres/SQL Server with minor changes).
-- Run:  python sql/run_queries.py

-- name: 01_kpis_by_app  | headline KPIs per app
SELECT a.app_name,
       COUNT(*)                                              AS reviews,
       ROUND(AVG(f.rating), 2)                               AS avg_rating,
       ROUND(100.0 * AVG(f.rating = 1), 1)                   AS pct_1_star,
       ROUND(100.0 * AVG(f.sentiment_label = 'Negative'), 1) AS pct_negative
FROM fact_reviews f JOIN dim_app a USING (app_key)
GROUP BY a.app_name
ORDER BY avg_rating;

-- name: 02_theme_impact  | rating gain if a theme were fixed (same logic as the DAX measure and the Python pipeline)
WITH app_tot AS (
    SELECT app_key, COUNT(*) AS n, SUM(rating) AS s FROM fact_reviews GROUP BY app_key),
theme AS (
    SELECT f.app_key, b.theme_key, COUNT(*) AS n_t, SUM(f.rating) AS s_t
    FROM fact_reviews f JOIN bridge_review_theme b USING (review_key)
    GROUP BY f.app_key, b.theme_key
    HAVING COUNT(*) >= 10)
SELECT a.app_name, d.theme_name, th.n_t AS reviews,
       ROUND(100.0 * th.n_t / t.n, 1)                                   AS share_pct,
       ROUND(1.0 * th.s_t / th.n_t, 2)                                  AS avg_rating_theme,
       ROUND(1.0 * th.n_t / t.n *
             MAX(1.0 * (t.s - th.s_t) / (t.n - th.n_t) - 1.0 * th.s_t / th.n_t, 0), 3) AS gain
FROM theme th
JOIN app_tot t USING (app_key) JOIN dim_app a USING (app_key) JOIN dim_theme d USING (theme_key)
ORDER BY a.app_name, gain DESC;

-- name: 03_top3_fixes_per_app  | window function: rank themes inside each app
WITH app_tot AS (SELECT app_key, COUNT(*) n, SUM(rating) s FROM fact_reviews GROUP BY app_key),
theme AS (
    SELECT f.app_key, b.theme_key, COUNT(*) n_t, SUM(f.rating) s_t
    FROM fact_reviews f JOIN bridge_review_theme b USING (review_key)
    GROUP BY f.app_key, b.theme_key HAVING COUNT(*) >= 10),
scored AS (
    SELECT th.app_key, th.theme_key,
           1.0 * th.n_t / t.n * MAX(1.0 * (t.s - th.s_t) / (t.n - th.n_t) - 1.0 * th.s_t / th.n_t, 0) AS gain
    FROM theme th JOIN app_tot t USING (app_key)),
ranked AS (SELECT *, RANK() OVER (PARTITION BY app_key ORDER BY gain DESC) AS rnk FROM scored)
SELECT a.app_name, r.rnk AS priority, d.theme_name, ROUND(r.gain, 3) AS gain
FROM ranked r JOIN dim_app a USING (app_key) JOIN dim_theme d USING (theme_key)
WHERE r.rnk <= 3 ORDER BY a.app_name, r.rnk;

-- name: 04_rolling_7_day_rating  | 7-day rolling average rating per app (weighted by review count)
WITH daily AS (
    SELECT app_key, date_key, COUNT(*) AS n, SUM(rating) AS s FROM fact_reviews GROUP BY app_key, date_key)
SELECT a.app_name, d.date, n AS reviews_that_day,
       ROUND(1.0 * SUM(s) OVER w / SUM(n) OVER w, 2) AS rolling_7d_avg_rating
FROM daily JOIN dim_app a USING (app_key) JOIN dim_date d USING (date_key)
WINDOW w AS (PARTITION BY app_key ORDER BY date_key ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
ORDER BY a.app_name, d.date DESC
LIMIT 12;

-- name: 05_spike_check  | recent 3 days vs previous 21 days per app x theme (SQL version of the z-test inputs)
WITH bounds AS (
    SELECT app_key, DATE(MAX(review_time), '+1 day') AS end_day FROM fact_reviews GROUP BY app_key),
tagged AS (
    SELECT f.app_key, f.review_key,
           CASE WHEN DATE(f.review_time) >= DATE(b.end_day, '-3 day') THEN 'recent'
                WHEN DATE(f.review_time) >= DATE(b.end_day, '-24 day') THEN 'baseline' END AS period
    FROM fact_reviews f JOIN bounds b USING (app_key)),
sizes AS (SELECT app_key, period, COUNT(*) AS n FROM tagged WHERE period IS NOT NULL GROUP BY app_key, period),
hits AS (
    SELECT t.app_key, t.period, br.theme_key, COUNT(*) AS k
    FROM tagged t JOIN bridge_review_theme br USING (review_key) WHERE t.period IS NOT NULL
    GROUP BY t.app_key, t.period, br.theme_key)
SELECT a.app_name, d.theme_name,
       ROUND(100.0 * hr.k / sr.n, 1) AS recent_share_pct,
       ROUND(100.0 * hb.k / sb.n, 1) AS baseline_share_pct,
       ROUND(100.0 * hr.k / sr.n - 100.0 * hb.k / sb.n, 1) AS change_pp
FROM hits hr
JOIN hits hb ON hb.app_key = hr.app_key AND hb.theme_key = hr.theme_key AND hb.period = 'baseline'
JOIN sizes sr ON sr.app_key = hr.app_key AND sr.period = 'recent'
JOIN sizes sb ON sb.app_key = hr.app_key AND sb.period = 'baseline'
JOIN dim_app a ON a.app_key = hr.app_key JOIN dim_theme d ON d.theme_key = hr.theme_key
WHERE hr.period = 'recent'
ORDER BY change_pp DESC LIMIT 8;

-- name: 06_most_upvoted_per_theme  | ROW_NUMBER: the review other customers agree with most, per theme
WITH r AS (
    SELECT d.theme_name, a.app_name, f.thumbs_up, f.rating, f.review_text,
           ROW_NUMBER() OVER (PARTITION BY d.theme_name ORDER BY f.thumbs_up DESC) AS rn
    FROM fact_reviews f
    JOIN bridge_review_theme b USING (review_key) JOIN dim_theme d USING (theme_key) JOIN dim_app a USING (app_key))
SELECT theme_name, app_name, thumbs_up, rating, SUBSTR(review_text, 1, 110) AS review
FROM r WHERE rn = 1 ORDER BY thumbs_up DESC;

-- name: 07_weekday_pattern  | does mood change by weekday?
SELECT d.weekday, ROUND(AVG(f.rating), 2) AS avg_rating, COUNT(*) AS reviews
FROM fact_reviews f JOIN dim_date d USING (date_key)
GROUP BY d.weekday, d.weekday_number ORDER BY d.weekday_number;

-- name: 08_data_quality_mismatch  | star rating vs model sentiment disagree: label noise check
SELECT COUNT(*) AS one_star_but_positive_sentiment
FROM fact_reviews WHERE rating = 1 AND sentiment_label = 'Positive';

-- name: 09_multi_theme_reviews  | how many complaints per review
SELECT theme_count AS themes_in_review, COUNT(*) AS reviews, ROUND(AVG(rating), 2) AS avg_rating
FROM fact_reviews GROUP BY theme_count ORDER BY theme_count;
