# RFM Cluster360 — Implementation Plan

> **Student Context:** 3rd-year B.Tech CS (Data Science), targeting cloud/data engineering roles.
> **Execution Model:** Gemini 3.8 Flash (High mode), phase-by-phase with git commits after each phase.
> **Repository:** `github.com/Khare69/RFM-Cluster360` (public)

---

## Table of Contents

1. [Phase 0 — Repository Bootstrap & Initial Push (Completed)](#phase-0)
2. [Phase 1 — Critical Bug Fixes: Runtime Crashes (Completed)](#phase-1)
3. [Phase 2 — Data Integrity & Cleaning Fixes (Completed)](#phase-2)
4. [Phase 3 — ML Pipeline Fixes: Persona Classification & Clustering (Completed)](#phase-3)
5. [Phase 4 — Visualization & UI Fixes (Completed)](#phase-4)
6. [Phase 5 — Cloud Feature A: S3/Cloud Storage Integration + Parquet (Completed)](#phase-5)
7. [Phase 6 — Cloud Feature B: Docker Containerization (Completed)](#phase-6)
8. [Phase 7 — Cloud Feature C: MLOps Model Artifact Registry (Completed)](#phase-7)
9. [Phase 8 — Cloud Feature D: GitHub Actions CI/CD Pipeline (Completed)](#phase-8)
10. [Phase 9 — Security & Critical Data Integrity Fixes](#phase-9)
11. [Phase 10 — Analytical Correctness & Persona Logic](#phase-10)
12. [Phase 11 — Architectural Decoupling & Pipeline Robustness](#phase-11)
13. [Phase 12 — Production Engineering & Validation](#phase-12)
14. [Dependency Summary](#dependencies)
15. [Execution Notes & Standard Phase Completion Protocol](#execution-notes)

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

### Task 1.5 — Documentation & Git Push
- Update `README.md` to document the 4 runtime crash fixes and test coverage.
- Stage all changes, commit with semantic message, and push to GitHub repository.

### Files Modified
- `components/filters.py`
- `src/ingestion.py`
- `app.py`
- `tests/test_ingestion.py` (new tests added)
- `README.md` (documentation update)

### Git Commit & Push
```bash
git add .
git commit -m "fix: resolve 4 critical runtime crashes — slider step overflow, duplicate column mapping, StringIO type error, and k-slider edge case"
git push origin main
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

### Task 2.4 — Documentation & Git Push
- Update `README.md` to document data integrity fixes (accounting negatives, ID normalization, null invoices) and itemized hygiene audit breakdown.
- Stage all changes, commit with semantic message, and push to GitHub repository.

### Files Modified
- `src/cleaning.py` (3 fixes + `CleaningReport` new field)
- `tests/test_cleaning.py` (3 new tests)
- `components/data_preview.py` (add new field to breakdown table)
- `README.md` (documentation update)

### Git Commit & Push
```bash
git add .
git commit -m "fix: data integrity — accounting parentheses, float string customer IDs, and null invoice_no validation"
git push origin main
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

### Task 3.3 — Documentation & Git Push
- Update `README.md` to document ML composite persona ranking, collision handling qualifiers for $k \ge 10$, and low-variance clustering guard.
- Stage all changes, commit with semantic message, and push to GitHub repository.

### Files Modified
- `src/personas.py` (label assignment refactor)
- `src/clustering.py` (variance guard)
- `pages/2_RFM_Explorer.py` (warning banner)
- `tests/test_personas.py` (updated + new tests)
- `tests/test_clustering.py` (new test)
- `README.md` (documentation update)

### Git Commit & Push
```bash
git add .
git commit -m "fix: ML pipeline — percentile-based persona ranking, label collision handling, and low-variance clustering guard"
git push origin main
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

### Task 4.4 — Documentation & Git Push
- Update `README.md` to document UI & visualization fixes (cohort heatmap contiguous reindexing, dynamic font contrast, zero-spend scatter safety, and `@st.cache_data` caching).
- Stage all changes, commit with semantic message, and push to GitHub repository.

### Files Modified
- `src/cohort.py`
- `viz/cohort_heatmap.py`
- `viz/overview_charts.py`
- `src/ingestion.py`, `src/rfm.py`, `src/clustering.py` (caching decorators)
- `tests/test_cohort.py` (updated test)
- `tests/test_rfm.py` (new test)
- `README.md` (documentation update)

### Git Commit & Push
```bash
git add .
git commit -m "fix: UI — cohort heatmap reindexing & text contrast, log-scale edge case, and @st.cache_data performance boost"
git push origin main
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

### Task 5.6 — Documentation & Git Push
- Update `README.md` with AWS S3 integration instructions, mock mode configuration, Parquet serialization guide, and test suite metrics.
- Stage all changes, commit with semantic message, and push to GitHub repository.

### Files Created/Modified
- `src/cloud_storage.py` (**new**)
- `src/export.py` (Parquet export function)
- `components/download_button.py` (format toggle)
- `app.py` (S3 ingestion tab)
- `.env.example` (new AWS variables)
- `requirements.txt` (add `boto3`, `moto`)
- `tests/test_cloud_storage.py` (**new**)
- `README.md` (documentation update)

### Git Commit & Push
```bash
git add .
git commit -m "feat: cloud storage — S3 bucket ingestion/export with local mock mode + Apache Parquet support"
git push origin main
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

### Task 6.4 — Documentation & Git Push

- Add "🐳 Run with Docker" section to `README.md`, including image build, docker-compose commands, port bindings, and health-check details.
- Stage all changes, commit with semantic message, and push to GitHub repository.

### Files Created/Modified
- `Dockerfile` (**new**)
- `docker-compose.yml` (**new**)
- `.dockerignore` (**new**)
- `README.md` (Docker documentation update)

### Git Commit & Push
```bash
git add .
git commit -m "feat: Docker containerization — multi-stage Dockerfile, compose config, and health checks"
git push origin main
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

### Task 7.5 — Documentation & Git Push
- Update `README.md` with MLOps Model Registry architecture, artifact directory structure, model versioning/comparison instructions, and test metrics.
- Stage all changes, commit with semantic message, and push to GitHub repository.

### Files Created/Modified
- `src/model_registry.py` (**new**)
- `pages/6_Model_Registry.py` (**new**)
- `app.py` (auto-save hook in `run_pipeline`)
- `artifacts/models/` (**new directory**, add to `.gitignore`)
- `.gitignore` (add `artifacts/models/`)
- `tests/test_model_registry.py` (**new**)
- `README.md` (documentation update)

### Git Commit & Push
```bash
git add .
git commit -m "feat: MLOps — model artifact versioning registry with save/load/compare and auto-persistence"
git push origin main
```

---

<a name="phase-8"></a>
## Phase 8 — Cloud Feature D: GitHub Actions CI/CD Pipeline (Completed)

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

### Task 8.4 — Documentation & Final Git Push

| Field | Details |
|-------|---------|
| **Modified File** | `README.md` |
| **Tasks** | Update `README.md` with CI workflow documentation, status badge, development vs production dependencies explanation, and final project verification. |
| **Action** | Stage all changes, commit with semantic message, and push to GitHub repository. |

### Files Created/Modified
- `.github/workflows/ci.yml` (**new**)
- `requirements-dev.txt` (**new**)
- `requirements.txt` (remove test deps)
- `tests/test_ci_config.py` (**new**)
- `README.md` (CI badge + workflow documentation)

### Git Commit & Push
```bash
git add .
git commit -m "ci: GitHub Actions CI/CD pipeline with automated pytest on push/PR + status badge"
git push origin main
```

---

---

<a name="phase-9"></a>
## Phase 9 — Security & Critical Data Integrity Fixes

**Goal:** Address immediate security vulnerabilities (path traversal), runtime crashes, and silent data corruption (ID truncation and out-of-memory risks).

### Task 9.1 — Path Traversal Vulnerability in Local Mock S3
- **File:** `src/cloud_storage.py`
- **Issue:** `get_mock_s3_path` allows `../` sequences, enabling arbitrary file read/write outside the `data/mock_s3/` directory.
- **Fix:** Use `pathlib.Path.resolve().is_relative_to()` to strictly enforce containment within the mock root directory. Raise `ValueError` if the path escapes.

### Task 9.2 — S3 Ingestion Memory & Size Safeguards
- **File:** `src/cloud_storage.py`
- **Issue:** Reading full S3 objects directly into memory (`response["Body"].read()`) can crash the application on large datasets.
- **Fix:** Inspect `ContentLength` in the S3 response metadata before reading. Implement a configurable size limit (e.g., 50MB) matching the CSV upload limit.

### Task 9.3 — Live Classifier Crash (Missing NumPy)
- **File:** `pages/6_Model_Registry.py`
- **Issue:** Clicking "Classify Customer Segment" raises `NameError: name 'np' is not defined` because numpy is used but not imported.
- **Fix:** Add `import numpy as np`. Include an automated Streamlit `AppTest` smoke test to catch page-level exceptions.

### Task 9.4 — Customer ID Normalization (Leading Zeros)
- **File:** `src/cleaning.py` (`_clean_id` and pandas CSV parsing)
- **Issue:** Converting IDs to floats/integers strips leading zeros (e.g., `"00123"` becomes `"123"`), incorrectly merging distinct customers.
- **Fix:** Force Pandas to read ID columns as strings (`dtype=str`) during ingestion. Only strip decimal `.0` suffixes, preserving original string contents and leading zeros.

### Task 9.5 — Documentation & Git Push
- Update `README.md` to document the security and data integrity fixes (mock S3 path traversal, S3 memory limit, missing imports, ID normalization).
- Stage all changes, commit with semantic message, and push to GitHub repository.

### Files Modified
- `src/cloud_storage.py`
- `pages/6_Model_Registry.py`
- `src/cleaning.py`
- `README.md` (documentation update)

### Git Commit & Push
```bash
git add .
git commit -m "fix: security & data integrity — mock S3 path traversal, S3 size limit, live classifier crash, ID normalization"
git push origin main
```

---

<a name="phase-10"></a>
## Phase 10 — Analytical Correctness & Persona Logic

**Goal:** Ensure the clustering and labeling algorithms are mathematically sound and reflect real-world customer behavior rather than arbitrary rankings.

### Task 10.1 — Variance Guard Pre-Scaling
- **File:** `src/clustering.py`
- **Issue:** The variance guard currently checks `scaled_data.std()`. Because `StandardScaler` forces variance to 1, the guard silently fails for near-constant data.
- **Fix:** Move the variance check to evaluate `log1p(data).std(axis=0)` *before* scaling.

### Task 10.2 — Rule-Based Persona Classification
- **File:** `src/personas.py`
- **Issue:** Personas are currently assigned by forcing clusters into a ranked list, meaning a low-value cluster could be labeled "Promising" just because of its relative rank.
- **Fix:** Implement explicit, behavior-based threshold rules (using absolute values or overall percentiles) to assign personas. If a cluster does not match a standard marketing persona, assign a neutral descriptive label (e.g., "Segment B — Recent, Low Spend").

### Task 10.3 — Explicit Return & Refund Policy (Cleaning Policy)
- **File:** `src/cleaning.py`
- **Issue:** All negative quantities/prices are discarded as cancelled. This ignores refunds and inflates net customer spend.
- **Fix:** Introduce a `CleaningPolicy` dataclass allowing analysts to toggle between "Gross Spend" (drop negatives) and "Net Spend" (sum negatives). Explicitly calculate refund rates independently.

### Task 10.4 — Configurable Currency Display
- **File:** Throughout UI and Exports
- **Issue:** Currency is hard-coded as `$` dollars, which is inaccurate for datasets in other currencies (e.g., £, €, ₹).
- **Fix:** Make currency a configuration option set during ingestion, propagating to chart labels, text metrics, and export file headers.

### Task 10.5 — Documentation & Git Push
- Update `README.md` to document the analytical improvements (variance guard pre-scaling, rule-based personas, net returns cleaning policy, configurable currency).
- Stage all changes, commit with semantic message, and push to GitHub repository.

### Files Modified
- `src/clustering.py`
- `src/personas.py`
- `src/cleaning.py`
- `README.md` (documentation update)
- UI and Export files (for configurable currency)

### Git Commit & Push
```bash
git add .
git commit -m "fix: analytical correctness — pre-scale variance guard, rule-based personas, net returns, configurable currency"
git push origin main
```

---

<a name="phase-11"></a>
## Phase 11 — Architectural Decoupling & Pipeline Robustness

**Goal:** Separate pure analytical logic from the Streamlit UI presentation layer, improving reusability, testability, and MLOps reliability.

### Task 11.1 — Remove UI Dependencies from Core Analytics
- **Files:** `src/rfm.py`, `src/clustering.py`, `src/ingestion.py`
- **Issue:** Core functions import `streamlit` and use `@st.cache_data`. This couples the business logic to the UI framework, preventing headless execution or API reuse.
- **Fix:** Remove all `streamlit` imports from `src/`. Move caching decorators to a UI orchestration layer (e.g., a new `application/segmentation_service.py` or wrapper functions in `app.py`).

### Task 11.2 — Unified scikit-learn Pipeline for Artifacts
- **File:** `src/model_registry.py` and `src/clustering.py`
- **Issue:** The registry currently saves K-Means and the Scaler as separate objects, while `log1p` transformations are hard-coded in the prediction function. This invites preprocessing drift.
- **Fix:** Bundle `FunctionTransformer(np.log1p)`, `StandardScaler()`, and `KMeans()` into a single `sklearn.pipeline.Pipeline` artifact.

### Task 11.3 — Explicit Model Saving
- **File:** `app.py`
- **Issue:** The pipeline automatically saves a model to the registry every time it runs, filling the registry with exploratory junk.
- **Fix:** Remove auto-save. Add an explicit "Save / Promote Model to Registry" button in the UI, requiring user intent to persist an artifact.

### Task 11.4 — Documentation & Git Push
- Update `README.md` to document the architecture changes (decoupled UI/analytics, sklearn pipeline artifact, explicit model saving).
- Stage all changes, commit with semantic message, and push to GitHub repository.

### Files Modified
- `src/rfm.py`, `src/clustering.py`, `src/ingestion.py`
- `src/model_registry.py`
- `app.py`
- `README.md` (documentation update)

### Git Commit & Push
```bash
git add .
git commit -m "refactor: architectural decoupling — remove Streamlit from core, unified sklearn pipeline, explicit model persistence"
git push origin main
```

---

<a name="phase-12"></a>
## Phase 12 — Production Engineering & Validation

**Goal:** Harden the CI/CD pipeline and introduce analytical baseline validation to prove the model's worth on real-world retail datasets.

### Task 12.1 — Expanded CI Quality Gates
- **File:** `.github/workflows/ci.yml`, `requirements-dev.txt`
- **Issue:** CI only runs `pytest`. Lacks linting, static type checking, and container verification.
- **Fix:** Add `ruff` (linting/formatting) and `pyright` (static typing) to the CI workflow. Add a step to build the Docker image in CI to guarantee the `Dockerfile` works.

### Task 12.2 — Analytical Baselines & Snapshot Dates
- **File:** `src/rfm.py`, `pages/2_RFM_Explorer.py`
- **Issue:** No clear indication of the data snapshot date, and clustering results are presented without a baseline comparison.
- **Fix:** Explicitly display the reference date (observation window) used for Recency calculations. Implement a simple quintile-based RFM scoring baseline alongside K-Means for interpretability comparisons.

### Task 12.3 — Documentation & Git Push
- Update `README.md` to document the expanded CI quality gates, baseline analytical metrics, and snapshot dates.
- Stage all changes, commit with semantic message, and push to GitHub repository.

### Files Modified
- `.github/workflows/ci.yml`
- `requirements-dev.txt`
- `src/rfm.py`, `pages/2_RFM_Explorer.py`
- `README.md` (documentation update)

### Git Commit & Push
```bash
git add .
git commit -m "ci: production engineering — ruff, pyright, docker build in CI, analytical baselines"
git push origin main
```

---

> **Post-MVP Roadmap (Deferred for later iterations):**
> *Segment migration over time, campaign outcome tracking, churn prediction, product affinity recommendations, and formal database integration (Delta Lake / BigQuery).*

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

```text
* ci: production engineering — ruff, pyright, docker build in CI, analytical baselines (Phase 12)
* refactor: architectural decoupling — remove Streamlit from core, unified sklearn pipeline, explicit model persistence (Phase 11)
* fix: analytical correctness — pre-scale variance guard, rule-based personas, net returns, configurable currency (Phase 10)
* fix: security & data integrity — mock S3 path traversal, S3 size limit, live classifier crash, ID normalization (Phase 9)
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

<a name="execution-notes"></a>
## Execution Notes & Standard Phase Completion Protocol

> [!IMPORTANT]
> **Mandatory Standard Closing Step for Every Phase:**
> After completing code implementation and passing all unit tests for any phase, the following closing sequence MUST be executed:
> 1. **Update `README.md` Side-by-Side:**
>    - Document all new features, bug fixes, architecture updates, test suite expansions, and roadmap progress.
>    - Keep badges, dependencies, and configuration instructions up to date.
> 2. **Run Full Test Suite:**
>    - Execute `pytest -v` to ensure 100% test pass rate with zero regressions.
> 3. **Stage All Changes:**
>    - `git add .` (including source code, unit tests, configuration files, and `README.md`).
> 4. **Commit with Semantic Commit Message:**
>    - Use conventional commit prefix (`feat:`, `fix:`, `docs:`, `ci:`, etc.) matching the phase specification.
> 5. **Push to Remote GitHub Repository:**
>    - `git push origin main` so the remote repository on GitHub is always completely synchronized with local progress.

> [!TIP]
> **Phases 0–8** (bootstrap, runtime fixes, data integrity, ML pipeline, visualization fixes, S3/Parquet cloud features, Docker containerization, MLOps model artifact registry, and GitHub Actions CI/CD pipeline) are **100% Complete** with all 53 automated tests passing across the entire project lifecycle with zero regressions.
> All production code, container specifications, and cloud integrations are fully implemented, tested, and synchronized.
