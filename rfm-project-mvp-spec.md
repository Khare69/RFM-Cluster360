# RFM Customer Segmentation Tool — Actual Build Plan

## What this is

A Streamlit app that takes a retail transaction CSV, calculates RFM (Recency, Frequency, Monetary) per customer, clusters customers with K-Means, and gives each cluster a human-readable label like "Champions" or "At Risk." You can search a customer and see their profile, and export the segmented list.

That's it. That's the project. Everything below is scoped to get this built and working well, not to list every feature a "customer intelligence platform" could theoretically have.

I cut the churn prediction, CLV modeling, market basket analysis, database layer, and the rest — not because they're bad ideas, but because most of them either need data you probably don't have (labeled churn outcomes), add complexity without adding much signal (dynamic weight sliders on top of K-Means), or are just scope creep dressed up as features (auth, multi-user, Docker deployment). If the core tool turns out well and you have time left, there's a "if you want more" list at the bottom — but build the core first and get it actually working before you touch any of that.

---

## Tech stack (no changes needed here, this part was fine)

- Python, pandas, numpy
- scikit-learn (StandardScaler, KMeans, silhouette_score)
- plotly (for the 3D scatter — matplotlib/seaborn can handle the static charts)
- Streamlit for the app itself
- pytest for testing
- Git/GitHub, obviously

Skip SQLite/Postgres for v1. CSV in, CSV out. A database is a good idea if this becomes a real multi-user product later — right now it's just extra plumbing that doesn't make the analysis better.

---

## Project structure

```
rfm-tool/
├── app.py                    # Streamlit entrypoint
├── requirements.txt
├── README.md
├── data/
│   └── sample_retail.csv     # so people can try it without their own data
├── src/
│   ├── ingestion.py          # load + validate uploaded file
│   ├── cleaning.py           # handle missing/bad rows
│   ├── rfm.py                # recency, frequency, monetary calc
│   ├── clustering.py         # scaling, kmeans, elbow/silhouette
│   ├── personas.py           # cluster centroid -> label
│   └── viz.py                # chart builders
└── tests/
    ├── test_rfm.py
    └── test_clustering.py
```

No `pages/` folder with ten dashboards yet. One app, a sidebar with maybe 4 views. Add more views later if the app actually needs them.

---

## The build, in the order I'd actually do it

### Step 1 — Get data in and cleaned (day 1-2)

- File uploader, CSV only for now (add Excel later if you care, it's a five-minute addition once the pipeline works)
- Try to auto-detect the standard columns (CustomerID, InvoiceDate, Quantity, UnitPrice, InvoiceNo) by matching common variants of the name. Don't overbuild this — a dictionary of aliases and a fuzzy string match is enough. Let the user manually fix anything it gets wrong.
- Clean the data and **show what you removed**, don't just silently drop rows:
  - missing CustomerID
  - cancelled orders (usually invoice numbers starting with "C")
  - negative or zero quantity
  - zero or negative price
- Show a simple before/after row count. That's your "data quality" feature — you don't need a fake percentage score with four sub-metrics, just tell the person what happened to their data honestly.

### Step 2 — RFM calculation (day 2-3)

- Recency: days since last purchase, relative to a snapshot date. Default the snapshot date to the day after the last transaction in the dataset, but let the user override it with a date picker. This part of the original doc was actually right and worth keeping — hardcoding "today" breaks the moment someone uploads a dataset from 2019.
- Frequency: count of *unique invoices* per customer, not row count. One order with 5 line items is one purchase, not five.
- Monetary: sum of (quantity × unit price) per customer.
- Write this as pure functions you can unit test against hand-calculated values on 3-4 known fake customers. Don't skip this — RFM logic has enough off-by-one traps (inclusive vs exclusive date ranges, whether returns net against monetary) that you want a test catching it, not a demo catching it.

### Step 3 — Clustering (day 3-5)

- Log-transform Frequency and Monetary (they're almost always right-skewed), then StandardScaler all three.
- Run K-Means for k = 2 through 8, plot the elbow curve and silhouette scores, pick the best k automatically (highest silhouette, with the elbow as a sanity check) but let the user override with a dropdown.
- That's enough. Skip DBSCAN and hierarchical for v1 — running one algorithm well beats running three algorithms and doing a shallow "comparison" between them. If you want to add DBSCAN later as a stretch goal, fine, but don't let "algorithm comparison" become a whole phase before you have one working segmentation.
- Skip the dynamic RFM weight sliders. Here's the honest reason: K-Means already clusters on all three dimensions, and manually reweighting scaled inputs before clustering is a fairly shallow gimmick to present as a headline feature. If you want it later as a "what happens if I emphasize recency" toy, it's cheap to bolt on — just don't build your project around it.

### Step 4 — Persona labeling (day 5)

- For each cluster, look at whether its centroid is high/low on each of R, F, M relative to the other clusters (e.g., top-third / middle-third / bottom-third).
- Map combinations to labels using simple rules:
  - Low recency, high frequency, high monetary → **Champions**
  - Low recency, high frequency, low-mid monetary → **Loyal Customers**
  - High recency, low frequency, low monetary → **Lost / Dormant**
  - Low recency, low frequency → **New Customers**
  - High recency, previously high frequency/monetary → **At Risk**
- Write these as an actual small rules table in code (a dict or a few if/elif branches), not something you hardcode per-dataset. That's the "configurable business rules" idea from the original doc, and it's a genuinely good one — just don't overbuild the config UI for it in v1.

### Step 5 — Dashboard (day 5-7)

Five views, sidebar navigation:

1. **Overview** — total customers, total revenue, cluster size bar chart, revenue-by-segment bar chart.
2. **RFM Explorer** — distributions of R, F, M (histograms), and the elbow/silhouette chart so people can see why k was chosen.
3. **3D Cluster Map** — plotly 3D scatter, R/F/M axes, colored by segment, hover shows customer ID and their numbers.
4. **Customer Search** — type a customer ID, see their R/F/M, segment, and a simple purchase timeline (just a scatter of invoice dates).
5. **Cohort Retention** — heatmap of retention by acquisition month. This one earns its place: it answers a question the other four views can't ("are customers we acquired in March sticking around better than the ones from January?"), and it isn't much extra work once you already have transaction dates.
   - Group customers by the month of their *first* purchase — that's their cohort.
   - For each cohort, calculate the % of that cohort still making purchases in month 1, month 2, month 3, etc. after acquisition.
   - Plot it as a heatmap: rows = cohort month, columns = months since acquisition, cell = retention %. Darker/lighter shading for higher/lower retention.
   - This is maybe 30-40 lines of pandas (a groupby + pivot table), not a separate subsystem. Put the logic in a `cohort.py` module alongside the others.

Add a filter sidebar (country, segment, revenue range) once these five views work — it's a small addition once the data's in a dataframe, not a separate phase.

### Step 6 — Export (day 7)

- "Download segmented customers" button — CSV with CustomerID, RFM values, segment label. That's the whole export feature. Skip PDF/Excel report generation for v1; it's a lot of formatting work for something a hiring manager or professor will glance at once.

### Step 7 — Tests + README + demo data (day 7-8)

- Unit tests for RFM math against hand-built fixtures, for clustering (assert k >= 2, no NaNs in output, correct number of labels), and for the cohort retention calc (a tiny fake dataset where you know the retention numbers by hand).
- A README that's honest about what it does and what it doesn't — don't oversell it as an "intelligence platform" in the README either. Say what it is: RFM segmentation with K-Means, an interactive dashboard, and a customer lookup.
- Include a small sample CSV so people can try it without hunting for the UCI Online Retail dataset themselves (though that dataset works well if you want a realistic one).

That's roughly a week and a half to two weeks of real work for one person (the cohort view adds maybe a day), and at the end you have something that actually runs end-to-end and that you can explain the internals of in an interview without hand-waving.

---

## If you finish this and want to keep going

Pick from here based on what you're actually interested in, not because a roadmap says so:

- **Churn risk score** (not "prediction" — be upfront that it's a rule-based deterioration score, e.g., current recency vs. typical purchase gap, unless you actually have historical labels to train on)
- **Basic CLV estimate** — AOV × frequency × an assumed lifespan window. Simple and honest, don't dress it up as more precise than it is.
- **One clean LLM call** that takes your computed segment/revenue numbers and writes a plain-English summary — this is a good differentiator and is maybe 20 lines of code, not a "Phase 37 AI Business Analyst" subsystem.

Everything past that (market basket analysis, anomaly detection, what-if simulator, auth, database, deployment) I'd genuinely leave out unless this stops being a portfolio project and becomes something you're shipping to real users.
