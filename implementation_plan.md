# RFM Cluster360 — Implementation Plan

> **Student Context:** 3rd-year B.Tech CS (Data Science), targeting cloud/data engineering roles.
> **Execution Model:** Gemini 3.8 Flash (High mode), phase-by-phase with git commits after each phase.
> **Repository:** `github.com/Khare69/RFM-Cluster360` (public)

---

## Table of Contents

1. [Phase 0 — Repository Bootstrap & Initial Push](#phase-0)
2. [Phase 1 — Critical Bug Fixes (Runtime Crashes)](#phase-1)
3. [Phase 2 — Data Integrity & Cleaning Fixes](#phase-2)
4. [Phase 3 — ML Pipeline Fixes (Persona Classification & Clustering)](#phase-3)
5. [Phase 4 — Visualization & UI Fixes](#phase-4)
6. [Phase 5 — Cloud Feature A: S3/Cloud Storage Integration + Parquet](#phase-5)
7. [Phase 6 — Cloud Feature B: Docker Containerization](#phase-6)
8. [Phase 7 — Cloud Feature C: MLOps Model Artifact Registry](#phase-7)
9. [Phase 8 — Cloud Feature D: GitHub Actions CI/CD Pipeline](#phase-8)
10. [Dependency Summary](#dependencies)

---

<a name="phase-0"></a>
## Phase 0 — Repository Bootstrap & Initial Push

**Goal:** Initialize git, create the public GitHub repository, push the entire working codebase as-is, and establish a professional README.

### Steps

| # | Task | Details |
|---|------|---------|
| 0.1 | `git init` inside project root | Initialize local repository |
| 0.2 | Review `.gitignore` | Ensure `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.vscode/` are excluded |
| 0.3 | `git add .` + `git commit` | Commit message: `feat: initial codebase — RFM Cluster360 MVP with Streamlit dashboard` |
| 0.4 | Create public GitHub repo via `gh repo create` | `gh repo create Khare69/RFM-Cluster360 --public --source=. --push` |
| 0.5 | Create professional `README.md` | Project overview, tech stack, architecture diagram (mermaid), screenshots placeholder, setup instructions |

### Files Created/Modified
- `README.md` (overwrite existing with professional version)

### Git Commit
```
feat: initial codebase — RFM Cluster360 MVP with Streamlit dashboard
```

### Dependencies
- None (existing packages only)

---

<a name="phase-1"></a>
## Phase 1 — Critical Bug Fixes (Runtime Crashes)

**Goal:** Fix 4 bugs that cause the app to crash with unhandled exceptions under specific input conditions.

### Bug 1.1 — Slider Step > Range Crash in Sidebar Filters

| Field | Details |
|-------|---------|
| **File** | `components/filters.py` (lines 61–68) |
| **Trigger** | When `max_spend - min_spend < 1.0` (e.g., filtered to customers with spend between $10.20 and $10.80), `step` is forced to `1.0` which exceeds the range. Streamlit raises `StreamlitAPIException`. |
| **Root Cause** | `step=max(1.0, ...)` enforces a minimum step of $1.00 regardless of the data range. |
| **Fix** | Clamp step to never exceed the actual range: `step = min(max(0.01, round(diff / 100, 2)), diff)` |
| **Test** | Manual: filter to a narrow monetary range < $1. Verify slider renders without crash. |

### Bug 1.2 — Duplicate Column Mapping Causes `KeyError` Crash

| Field | Details |
|-------|---------|
| **File** | `src/ingestion.py` → `validate_mapping()` (lines 168–190) |
| **Trigger** | User maps the same CSV column (e.g., `"OrderID"`) to both `customer_id` and `invoice_no`. `clean_transactions()` then crashes with `KeyError: 'customer_id'`. |
| **Root Cause** | `validate_mapping()` only checks for `None` values, not for duplicate column assignments. The `inv_map` dict in `cleaning.py` silently overwrites one key. |
| **Fix** | Add duplicate detection in `validate_mapping()`: check that all non-None values in the mapping are unique. Return descriptive error messages for duplicates. |
| **Test** | New unit test: `test_validate_mapping_duplicate_columns()` in `tests/test_ingestion.py`. |

### Bug 1.3 — `load_csv()` Crashes on `io.StringIO` Input

| Field | Details |
|-------|---------|
| **File** | `src/ingestion.py` → `load_csv()` (lines 63–82) |
| **Trigger** | Passing an `io.StringIO` object (e.g., from test fixtures). `.read()` returns a `str`, but `io.BytesIO(raw_bytes)` expects `bytes`. Raises `TypeError: a bytes-like object is required, not 'str'`. |
| **Root Cause** | No type check after reading from file-like objects. |
| **Fix** | After `raw_bytes = file_source.read()`, add: `if isinstance(raw_bytes, str): raw_bytes = raw_bytes.encode("utf-8")` |
| **Test** | New unit test: `test_load_csv_from_string_io_object()` in `tests/test_ingestion.py`. |

### Bug 1.4 — Sidebar k-Slider Crash When `min_value == max_value`

| Field | Details |
|-------|---------|
| **File** | `app.py` (lines 148–154) |
| **Trigger** | Dataset with only 3 customers → `max_value = min(8, max(2, 3-1)) = 2` and `min_value = 2`. Streamlit raises `StreamlitAPIException: Slider min and max cannot be equal`. |
| **Root Cause** | No guard for edge case where computed min equals max. |
| **Fix** | Check `if min_k_val < max_k_val:` before rendering slider. Otherwise display a static info badge: `st.sidebar.info(f"k is fixed at {min_k_val} for this dataset size")`. |
| **Test** | Manual: load a dataset with exactly 3 customers. |

### Files Modified
- `components/filters.py`
- `src/ingestion.py`
- `app.py`
- `tests/test_ingestion.py` (new tests added)

### Git Commit
```
fix: resolve 4 critical runtime crashes — slider step overflow, duplicate column mapping, StringIO type error, and k-slider edge case
```

---

<a name="phase-2"></a>
## Phase 2 — Data Integrity & Cleaning Fixes

**Goal:** Fix 3 data corruption bugs that silently produce incorrect customer profiles and skewed clustering.

### Bug 2.1 — Accounting Parentheses Invert Refund Quantities

| Field | Details |
|-------|---------|
| **File** | `src/cleaning.py` → `_to_clean_numeric()` (lines 107–111) |
| **Trigger** | Retail exports (SAP, QuickBooks) represent negative values as `(10.50)` instead of `-10.50`. The regex `r"[^\d.-]"` strips parentheses, turning `(10.50)` into positive `10.50`. Refunds are counted as sales. |
| **Fix** | Before stripping non-numeric chars, detect and convert accounting format: `s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)` |
| **Test** | New unit test: `test_clean_transactions_with_accounting_negatives()` in `tests/test_cleaning.py`. |

### Bug 2.2 — String-Float Customer IDs Split Single Customers

| Field | Details |
|-------|---------|
| **File** | `src/cleaning.py` → `_clean_id()` (lines 99–103) |
| **Trigger** | CSV contains mixed types: `1001.0` (float from pandas) and `"1001.0"` (string). `_clean_id("1001.0")` returns `"1001.0"` (unchanged), while `_clean_id(1001.0)` returns `"1001"`. Same customer gets split into two profiles. |
| **Fix** | Try float conversion on string values: `f = float(s); if f.is_integer(): return str(int(f))` |
| **Test** | New unit test: `test_clean_id_normalizes_float_strings()` in `tests/test_cleaning.py`. |

### Bug 2.3 — Null/Empty `invoice_no` Creates Phantom Customers

| Field | Details |
|-------|---------|
| **File** | `src/cleaning.py` (no validation for `invoice_no`) and `src/rfm.py` (line 116) |
| **Trigger** | Rows with `NaN` or empty `invoice_no` pass through cleaning. In RFM calculation, `nunique()` ignores NaN values, producing customers with `frequency = 0` but positive monetary spend. |
| **Fix** | Add explicit null/empty `invoice_no` filter in `clean_transactions()`, after the cancelled order filter. Track count in `CleaningReport`. |
| **Test** | New unit test: `test_clean_transactions_null_invoice_no()` in `tests/test_cleaning.py`. |

### Files Modified
- `src/cleaning.py` (3 fixes + `CleaningReport` new field)
- `tests/test_cleaning.py` (3 new tests)
- `components/data_preview.py` (add new field to breakdown table)

### Git Commit
```
fix: data integrity — accounting parentheses, float string customer IDs, and null invoice_no validation
```

---

<a name="phase-3"></a>
## Phase 3 — ML Pipeline Fixes (Persona Classification & Clustering)

**Goal:** Fix the persona mislabeling logic and improve the robustness of clustering evaluation.

### Bug 3.1 — Centroid Min-Max Normalization Distorts Segment Labels

| Field | Details |
|-------|---------|
| **File** | `src/personas.py` → `generate_cluster_label_mapping()` (lines 155–211) |
| **Problem** | Min-max normalization across only `k` centroid values (e.g., 3 points) forces the minimum centroid to `0.0` and maximum to `1.0`. A cluster with 5 orders is normalized to `0.16` because another cluster has 25 orders, mislabeling moderate-frequency customers as "New Customers (1-2 orders)". |
| **Fix** | **Rank clusters by composite score before assigning labels.** Sort centroids by a weighted composite (`0.35*R_inverted + 0.35*F + 0.30*M`) from best to worst. Assign labels in order from the predefined persona list (Champions → Loyal → ... → Lost/Dormant). This guarantees the strongest cluster always gets the top label and the weakest gets the bottom. |
| **Also Fix** | Label collision when `k >= 10` (only 9 labels). Add ordinal qualifiers (e.g., `"Needs Attention (Tier 2)"`) for overflow. |
| **Test** | Update `tests/test_personas.py` — add test for `k=10` ensuring all labels are unique. |

### Bug 3.2 — Clustering Evaluation on Identical/Near-Identical Data

| Field | Details |
|-------|---------|
| **File** | `src/clustering.py` → `evaluate_clusters()` (lines 63–134) |
| **Problem** | If all customers have nearly identical RFM values (e.g., a loyalty program with uniform spend), K-Means produces `ConvergenceWarning` spam and returns `best_k=2` with `silhouette=0.0`. No user-facing feedback. |
| **Fix** | Before clustering, check variance. If `scaled_data.std() < 0.01`, return early with a descriptive message instead of blindly fitting K-Means. Emit a warning to the user via the returned metrics dict. |
| **Test** | New unit test: `test_evaluate_clusters_low_variance()` in `tests/test_clustering.py`. |

### Files Modified
- `src/personas.py` (label assignment refactor)
- `src/clustering.py` (variance guard)
- `tests/test_personas.py` (updated + new tests)
- `tests/test_clustering.py` (new test)

### Git Commit
```
fix: ML pipeline — percentile-based persona ranking, label collision handling, and low-variance clustering guard
```

---

<a name="phase-4"></a>
## Phase 4 — Visualization & UI Fixes

**Goal:** Fix cohort heatmap readability, log-scale scatter edge case, and add `@st.cache_data` for performance.

### Bug 4.1 — Cohort Heatmap Missing Months & Illegible Text

| Field | Details |
|-------|---------|
| **File** | `src/cohort.py` (lines 58–72) and `viz/cohort_heatmap.py` (lines 61–74) |
| **Problem 1** | If a cohort has no activity in Month 1 or 2 but reappears in Month 4, the pivot table skips columns (M0 → M4). Heatmap shows a misleading jump. |
| **Problem 2** | White font (`color="white"`) on pale blue (low retention cells 0–20%) is invisible. |
| **Fix 1** | Reindex pivot table columns to `range(0, max_index + 1)` and fill missing months with `NaN`. |
| **Fix 2** | Use dynamic text color: dark (`#1A202C`) for cells below 50% retention, white for cells above 50%. |
| **Test** | Update `tests/test_cohort.py` — test with non-contiguous months. |

### Bug 4.2 — `rfm_summary_scatter` Log Scale with $0 Monetary

| Field | Details |
|-------|---------|
| **File** | `viz/overview_charts.py` → `rfm_summary_scatter()` (line 145) |
| **Problem** | `log_y=True` with monetary values of `$0.00` produces `-inf` on the y-axis. Points with $0 spend disappear from the chart. |
| **Fix** | Clip monetary to `max(0.01, monetary)` before plotting, or use `symlog` scale. |

### Enhancement 4.3 — Add `@st.cache_data` for Performance

| Field | Details |
|-------|---------|
| **File** | `src/ingestion.py`, `src/rfm.py`, `src/clustering.py` |
| **Problem** | Every Streamlit widget interaction re-runs the entire script. CSV re-reading and 70+ K-Means fits execute on every sidebar toggle. |
| **Fix** | Wrap pure functions with `@st.cache_data`: `load_csv`, `calculate_rfm`, `preprocess_rfm`, `evaluate_clusters`. Cache-busting occurs automatically when inputs change. |
| **Caveat** | Functions must accept hashable arguments only. `pd.DataFrame` is hashable in Streamlit's caching. |

### Files Modified
- `src/cohort.py`
- `viz/cohort_heatmap.py`
- `viz/overview_charts.py`
- `src/ingestion.py`, `src/rfm.py`, `src/clustering.py` (caching decorators)
- `tests/test_cohort.py` (updated test)

### Git Commit
```
fix: UI — cohort heatmap reindexing & text contrast, log-scale edge case, and @st.cache_data performance boost
```

---

<a name="phase-5"></a>
## Phase 5 — Cloud Feature A: S3/Cloud Storage Integration + Parquet

**Goal:** Add the ability to ingest data from AWS S3 buckets and export results in Apache Parquet format.

### New Dependencies
```
boto3>=1.28.0
moto>=4.2.0    # AWS mock for offline testing — dev dependency only
pyarrow>=14.0.0  # Already installed in your venv (v25.0.1)
```

### Task 5.1 — S3 Ingestion Module

| Field | Details |
|-------|---------|
| **New File** | `src/cloud_storage.py` |
| **Functionality** | `load_from_s3(bucket, key, aws_access_key_id, aws_secret_access_key, region)` — Downloads a CSV or Parquet file from an S3 bucket into a DataFrame. Falls back to a local mock directory (`data/mock_s3/`) if no credentials are provided. |
| **Auth Pattern** | Reads from environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) via `.env`. Never hardcoded. |
| **Mock Mode** | If `USE_LOCAL_MOCK=true` in `.env`, reads from `data/mock_s3/{bucket}/{key}` on disk instead of real AWS. This lets you demo the feature without an AWS account. |

### Task 5.2 — Parquet Export Module

| Field | Details |
|-------|---------|
| **Modified File** | `src/export.py` |
| **New Function** | `to_parquet_bytes(df) -> bytes` — Serializes DataFrame to Parquet binary via `pyarrow`. |
| **UI Integration** | Add a format toggle (CSV vs. Parquet) in `components/download_button.py`. |

### Task 5.3 — S3 Export (Upload Results Back to Cloud)

| Field | Details |
|-------|---------|
| **Modified File** | `src/cloud_storage.py` |
| **New Function** | `upload_to_s3(df, bucket, key, format="parquet")` — Uploads the segmented DataFrame to S3 as CSV or Parquet. |
| **UI Integration** | Add an "Export to S3" button in the export section (visible only when AWS credentials are configured). |

### Task 5.4 — UI Tab for Cloud Ingestion

| Field | Details |
|-------|---------|
| **Modified File** | `app.py` |
| **Change** | Add a third tab alongside "Upload CSV" and "Sample Dataset": `☁️ Load from S3 Bucket`. Renders input fields for Bucket Name, Object Key, and optional credentials. |

### Task 5.5 — Tests

| Field | Details |
|-------|---------|
| **New File** | `tests/test_cloud_storage.py` |
| **Tests** | `test_load_from_s3_mock()` using `moto` to mock AWS S3. `test_upload_to_s3_mock()`. `test_load_parquet_roundtrip()`. |

### Files Created/Modified
- `src/cloud_storage.py` (**new**)
- `src/export.py` (Parquet export function)
- `components/download_button.py` (format toggle)
- `app.py` (S3 ingestion tab)
- `.env.example` (new AWS variables)
- `requirements.txt` (add `boto3`, `moto`)
- `tests/test_cloud_storage.py` (**new**)

### Git Commit
```
feat: cloud storage — S3 bucket ingestion/export with local mock mode + Apache Parquet support
```

---

<a name="phase-6"></a>
## Phase 6 — Cloud Feature B: Docker Containerization

**Goal:** Package the entire application into a production-ready Docker container.

### New Dependencies
- Docker Desktop (must be installed on your machine — not a Python package)

### Task 6.1 — Multi-Stage Dockerfile

| Field | Details |
|-------|---------|
| **New File** | `Dockerfile` |
| **Strategy** | Multi-stage build: Stage 1 installs dependencies into a virtual layer. Stage 2 copies only the app code + installed packages (smaller final image). |
| **Base Image** | `python:3.11-slim` (lightweight, ~150MB vs ~900MB for full). |
| **Exposed Port** | `8501` (Streamlit default). |
| **Health Check** | `HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1` |

### Task 6.2 — Docker Compose for Local Development

| Field | Details |
|-------|---------|
| **New File** | `docker-compose.yml` |
| **Services** | Single service `app` that builds from `Dockerfile`, maps port `8501:8501`, mounts `.env` for configuration, and optionally mounts a `data/` volume for persistent uploads. |
| **Usage** | `docker compose up --build` |

### Task 6.3 — `.dockerignore`

| Field | Details |
|-------|---------|
| **New File** | `.dockerignore` |
| **Excludes** | `.venv/`, `__pycache__/`, `.git/`, `.env`, `tests/`, `.pytest_cache/`, `.vscode/` |

### Task 6.4 — Update README with Docker Instructions

- Add "🐳 Run with Docker" section to `README.md`.

### Files Created/Modified
- `Dockerfile` (**new**)
- `docker-compose.yml` (**new**)
- `.dockerignore` (**new**)
- `README.md` (Docker section added)

### Git Commit
```
feat: Docker containerization — multi-stage Dockerfile, compose config, and health checks
```

---

<a name="phase-7"></a>
## Phase 7 — Cloud Feature C: MLOps Model Artifact Registry

**Goal:** Persist trained K-Means models, scalers, and persona metadata as versioned artifacts with a UI for browsing and reloading past models.

### New Dependencies
```
joblib>=1.3.0    # Already installed in your venv (v1.5.3)
```

### Task 7.1 — Model Serialization Module

| Field | Details |
|-------|---------|
| **New File** | `src/model_registry.py` |
| **Functions** | `save_model_artifact(model, scaler, cluster_map, eval_metrics, save_dir="artifacts/models/") -> str` — Saves: (1) `kmeans_model.joblib` via joblib, (2) `scaler.joblib`, (3) `metadata.json` with timestamp, k, silhouette score, feature names, persona mapping. Returns the versioned directory path (e.g., `artifacts/models/v3_k4_20261005/`). |
| | `list_model_versions(save_dir) -> List[dict]` — Lists all saved model versions with their metadata. |
| | `load_model_artifact(version_dir) -> Tuple[KMeans, StandardScaler, dict, dict]` — Loads a saved model + scaler + metadata. |
| | `predict_segment(model, scaler, cluster_map, rfm_values) -> dict` — Predicts segment for new customer data using a saved model. |

### Task 7.2 — Model Registry UI Page

| Field | Details |
|-------|---------|
| **New File** | `pages/6_Model_Registry.py` |
| **Features** | Table of all saved model versions. "Save Current Model" button. "Load & Apply" button to restore a previous model. Side-by-side comparison of silhouette scores across versions. |

### Task 7.3 — Auto-Save After Pipeline Run

| Field | Details |
|-------|---------|
| **Modified File** | `app.py` → `run_pipeline()` |
| **Change** | After clustering completes, automatically save the model artifact. Store the version path in `st.session_state.model_version`. |

### Task 7.4 — Tests

| Field | Details |
|-------|---------|
| **New File** | `tests/test_model_registry.py` |
| **Tests** | `test_save_and_load_roundtrip()`, `test_list_versions()`, `test_predict_segment()`. |

### Files Created/Modified
- `src/model_registry.py` (**new**)
- `pages/6_Model_Registry.py` (**new**)
- `app.py` (auto-save hook in `run_pipeline`)
- `artifacts/models/` (**new directory**, add to `.gitignore`)
- `.gitignore` (add `artifacts/models/`)
- `tests/test_model_registry.py` (**new**)

### Git Commit
```
feat: MLOps — model artifact versioning registry with save/load/compare and auto-persistence
```

---

<a name="phase-8"></a>
## Phase 8 — Cloud Feature D: GitHub Actions CI/CD Pipeline

**Goal:** Automate testing on every push/PR with a GitHub Actions workflow. Add a status badge to the README.

### Task 8.1 — CI Workflow File

| Field | Details |
|-------|---------|
| **New File** | `.github/workflows/ci.yml` |
| **Triggers** | `push` to `main` branch, `pull_request` to `main` branch. |
| **Steps** | 1. Checkout code. 2. Set up Python 3.11. 3. Install dependencies from `requirements.txt`. 4. Run `pytest -v --tb=short`. 5. (Optional) Upload test results as artifact. |
| **Matrix** | Test on `ubuntu-latest` (standard for CI). |

### Task 8.2 — README Badge

| Field | Details |
|-------|---------|
| **Modified File** | `README.md` |
| **Change** | Add CI status badge at the top: `![CI](https://github.com/Khare69/RFM-Cluster360/actions/workflows/ci.yml/badge.svg)` |

### Task 8.3 — Requirements Split (Dev vs. Production)

| Field | Details |
|-------|---------|
| **New File** | `requirements-dev.txt` |
| **Contents** | `pytest>=7.4.0`, `moto>=4.2.0` (test-only dependencies). |
| **Modified File** | `requirements.txt` — remove `pytest`, keep only production deps. |
| **CI Config** | Install both: `pip install -r requirements.txt -r requirements-dev.txt` |

### Files Created/Modified
- `.github/workflows/ci.yml` (**new**)
- `requirements-dev.txt` (**new**)
- `requirements.txt` (remove test deps)
- `README.md` (badge + final polish)

### Git Commit
```
ci: GitHub Actions CI/CD pipeline with automated pytest on push/PR + status badge
```

---

<a name="dependencies"></a>
## Dependency Summary

### Production Dependencies (`requirements.txt` — final state)

| Package | Version | Purpose | Phase Added |
|---------|---------|---------|:-----------:|
| `pandas` | `>=2.0.0` | Core data manipulation | Existing |
| `numpy` | `>=1.24.0` | Numerical operations | Existing |
| `scikit-learn` | `>=1.3.0` | K-Means clustering, silhouette score | Existing |
| `plotly` | `>=5.17.0` | Interactive visualizations | Existing |
| `streamlit` | `>=1.28.0` | Web application framework | Existing |
| `python-dotenv` | `>=1.0.0` | Environment variable loading | Existing |
| `boto3` | `>=1.28.0` | AWS S3 SDK | Phase 5 |
| `pyarrow` | `>=14.0.0` | Apache Parquet read/write | Phase 5 |
| `joblib` | `>=1.3.0` | Model serialization (already installed via scikit-learn) | Phase 7 |

### Development Dependencies (`requirements-dev.txt`)

| Package | Version | Purpose | Phase Added |
|---------|---------|---------|:-----------:|
| `pytest` | `>=7.4.0` | Unit testing framework | Phase 8 (moved) |
| `moto` | `>=4.2.0` | AWS S3 mock for offline testing | Phase 5 |

### System Dependencies (Not Python packages)

| Tool | Purpose | Phase Needed |
|------|---------|:------------:|
| `git` | Version control | Phase 0 |
| `gh` (GitHub CLI) | Create remote repo from terminal | Phase 0 |
| Docker Desktop | Container builds (optional, for Phase 6 only) | Phase 6 |

---

## Git Commit History (Expected After All Phases)

```
* ci: GitHub Actions CI/CD pipeline with automated pytest on push/PR + status badge          (Phase 8)
* feat: MLOps — model artifact versioning registry with save/load/compare and auto-persistence (Phase 7)
* feat: Docker containerization — multi-stage Dockerfile, compose config, and health checks   (Phase 6)
* feat: cloud storage — S3 bucket ingestion/export with local mock mode + Parquet support     (Phase 5)
* fix: UI — cohort heatmap reindexing & text contrast, log-scale edge case, @st.cache_data   (Phase 4)
* fix: ML pipeline — percentile-based persona ranking, label collision, low-variance guard    (Phase 3)
* fix: data integrity — accounting parens, float string IDs, null invoice_no validation       (Phase 2)
* fix: resolve 4 critical runtime crashes — slider step, duplicate mapping, StringIO, k-slider (Phase 1)
* feat: initial codebase — RFM Cluster360 MVP with Streamlit dashboard                       (Phase 0)
```

---

## Execution Notes

> [!IMPORTANT]
> **After each phase:**
> 1. All modified/new files are committed to git.
> 2. Changes are pushed to `github.com/Khare69/RFM-Cluster360`.
> 3. `README.md` is updated with a changelog entry for that phase.
> 4. Tests are run (`pytest -v`) to verify no regressions.

> [!TIP]
> **Phases 1–4** (bug fixes) can be done in a single focused session (~2-3 hours).
> **Phases 5–8** (cloud features) each take ~1-2 hours individually.
> Total estimated time: **8-12 hours** across 2-3 sessions.
