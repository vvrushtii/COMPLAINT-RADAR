# Build the Power BI dashboard (≈ 40 minutes)

Everything except the final drag-and-drop is already done: the data is modelled as a star schema, the DAX is written and its logic was
cross-checked against the Python pipeline, and a matching dark theme is ready. You only assemble the visuals.

> **Honesty note:** I could not open Power BI Desktop where this was prepared, so the DAX has not been executed there. The *logic* was
> verified by re-implementing it in SQL (`sql/queries.sql`), which matches the Python results on all 40 app × theme rows. If Desktop shows an
> error on a measure, send me the exact message and I'll fix it.

## 0. Install
Power BI Desktop is free: Microsoft Store → search "Power BI Desktop" → Install. (Windows only.)

## 1. Load the data (5 min)
Home → **Get data → Text/CSV**, one file at a time from `powerbi/data/`, click **Load** each time:
`dim_app, dim_theme, dim_date, fact_reviews, bridge_review_theme, fact_impact, fact_alerts, model_results`

Then check data types in **Table view** (click a column → Column tools):
| Table | Column | Type |
|---|---|---|
| dim_date | date | Date |
| fact_reviews | review_time | Date/Time |
| fact_impact | first_date, last_date | Date |
| fact_alerts | is_spike | True/False (if it loaded as text: Transform → Replace values, then set type) |
| fact_reviews | review_key | Text |

Select `dim_date` → Table tools → **Mark as date table** → column `date`.

## 2. Relationships (5 min): Model view
Drag column onto column. Every relationship is **Many-to-one (*:1)**, **single** direction, from the fact/bridge (many) to the dimension (one):

| From (many) | To (one) |
|---|---|
| fact_reviews[app_key] | dim_app[app_key] |
| fact_reviews[date_key] | dim_date[date_key] |
| bridge_review_theme[review_key] | fact_reviews[review_key] |
| bridge_review_theme[theme_key] | dim_theme[theme_key] |
| fact_impact[app_key] | dim_app[app_key] |
| fact_impact[theme_key] | dim_theme[theme_key] |
| fact_alerts[app_key] | dim_app[app_key] |
| fact_alerts[theme_key] | dim_theme[theme_key] |

`model_results` stays unconnected. Arrange the diagram so dimensions are on top and facts below (this "looks like a star" in screenshots).

**Why a bridge table?** A review can mention several complaint themes, which is a many-to-many relationship. The standard fix is a bridge table
holding one row per (review, theme). The measures use `CROSSFILTER(..., BOTH)` to push a theme filter through the bridge into the reviews.

## 3. Measures (5 min)
Home → **Enter data** → rename the table `_Measures` → Load. Then for each measure in `measures.dax`: select `_Measures` → **New measure** → paste.
Apply the format strings listed at the bottom of that file.

## 4. Theme
View → **Themes → Browse for themes** → `ComplaintRadar-Dark.json`. (If Desktop rejects it, use any dark built-in theme and set the
page background to `#0F1117`; nothing else depends on it.)

## 5. Pages
Set every page to 16:9. Add a **slicer for dim_app[app_name]** (tile style) at the top of each page and use **View → Sync slicers** so it's shared.

### Page 1: Overview
| Visual | Fields |
|---|---|
| 5 × **Card** | Total Reviews · Avg Rating · 1-Star % · Negative % · Spike Status |
| **Card** (wide) | `#1 Problem` with subtitle `#1 Problem Gain` ("+x★ if fixed") |
| **Line chart** | X: dim_date[date] · Y: Rolling 7D Avg Rating · Legend: dim_app[app_name] |
| **Clustered bar** | Y: dim_theme[theme_name] · X: Rating Gain if Fixed · sorted descending (top 5 via Filters: Top N) |

### Page 2: Fix-first (the main page)
| Visual | Fields / format |
|---|---|
| **Clustered bar** | Y: dim_theme[theme_name] · X: **Gain (Pipeline)**. Format → **Error bars** → *Upper* = Gain CI High, *Lower* = Gain CI Low. Sort descending. Needs ONE app selected. |
| **Table** | theme_name, Theme Reviews, Theme Share %, Avg Rating (Theme), Rating Gain if Fixed, Fix Priority. Conditional-format Fix Priority with data bars. |
| **Text box** | "Impact = share of reviews in the theme × how much lower their rating is than everyone else's. Bars show 95% confidence ranges." |

### Page 3: Competitor benchmark
| Visual | Fields / format |
|---|---|
| **Matrix** | Rows: dim_app[app_name] · Columns: dim_theme[theme_name] · Values: Theme Share %. Cell elements → **Background color → Gradient** (white → red). This is the heatmap. |
| **Clustered column** | X: app_name · Y: Avg Rating, 1-Star %, Negative % |
| **Table** | app_name, Total Reviews, Avg Rating, 1-Star %, Negative % |
**Caveat to show on the page (text box):** apps cover different date ranges, so compare using the date slicer to a window all apps share.

### Page 4: Alerts & trends
| Visual | Fields |
|---|---|
| **Card** | Spike Status |
| **Table** | fact_alerts: app, theme_name, recent_share_pct, baseline_share_pct, change_pp, z. Filter `is_spike` = True when any exist. Conditional-format `change_pp` red/green. |
| **Line chart** | X: dim_date[date] · Y: Theme Share Rolling 7D · Legend: dim_theme[theme_name] (slicer to pick themes) |
| **Column chart** | X: dim_date[weekday] · Y: Avg Rating (sort by weekday_number) |

### Page 5: Voice of customer
| Visual | Fields |
|---|---|
| **Slicer** | dim_theme[theme_name] |
| **Table** | fact_reviews: review_text, rating, thumbs_up, sentiment_label. Sort by thumbs_up desc, turn on **word wrap**. In the Filters pane add the measure **Show Review**, set it to *is 1*. Now the theme slicer keeps only reviews that mention that theme. |

### Page 6: Model quality
| Visual | Fields |
|---|---|
| **Clustered bar** | Y: model_results[model] · X: Model Score · Legend: metric · **Slicer** on task (Sentiment / Themes) |
| **Text box** | "Sentiment: multilingual model 93% vs VADER 85%. Themes: supervised model barely beats keyword rules (0.70 vs 0.69), so I report it as a tie." |

## 6. Polish (this is what makes it look professional)
- **Titles:** every visual gets a title that states the *insight*, e.g. "Customer support is the biggest rating drag" instead of "Bar chart".
- **Navigation:** Insert → Buttons → Navigator → **Page navigator** at the top of each page.
- **Tooltips:** hover on theme bars should show Theme Reviews and Avg Rating (Theme): add them under *Tooltips* in the visual's field well.
- **Alignment:** select visuals → Format → Align / Distribute. Keep 16 px gaps and one font.
- **Colours:** keep brand colours for apps (Swiggy orange, Zomato red, Blinkit yellow, Zepto purple), set by the theme's first four data colours.
- **Mobile layout:** View → Mobile layout (optional, 5 minutes).

## 7. Check your work: you should see exactly these numbers
(All apps, the data as exported on the day this guide was written; they change when the daily job adds reviews and you re-export.)
| App | Reviews | Avg rating | 1-star % | Negative % |
|---|---|---|---|---|
| Swiggy | 2,057 | 2.52 | 56.1 | 62.7 |
| Zepto | 3,052 | 2.66 | 54.2 | 62.2 |
| Zomato | 1,647 | 2.97 | 44.2 | 53.4 |
| Blinkit | 2,291 | 3.57 | 28.3 | 39.6 |

And with **Swiggy** selected: Fix-first #1 = Customer support, **Rating Gain if Fixed = 0.284**, #2 Late / slow delivery 0.278, #3 Refund / money issues 0.210.
If those match, your relationships and measures are correct.

## 8. Refreshing the data
Re-run `python -m radar.export_powerbi` after the daily job, then in Power BI: Home → **Refresh**. (Files are overwritten in place.)

## 9. Export for LinkedIn / README
File → **Export → PDF**, or screenshot each page (Win+Shift+S). Put images in `powerbi/screenshots/`.

## What to say in an interview
- "I modelled it as a **star schema** with a **bridge table** because a review can mention several complaint themes."
- "The ranking metric is a **DAX measure**; I verified it against a SQL implementation and my Python pipeline, and all three agree."
- "I used **precomputed bootstrap confidence intervals** for the error bars rather than recomputing statistics in DAX, because that's the right tool for each job."
- "The same data powers a Streamlit app, a Power BI report and a SQL analysis layer."
