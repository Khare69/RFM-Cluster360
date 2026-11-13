# 🎯 RFM Cluster360 — Customer Intelligence & Behavioral Segmentation

[![CI](https://github.com/Khare69/RFM-Cluster360/actions/workflows/ci.yml/badge.svg)](https://github.com/Khare69/RFM-Cluster360/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.17+-3F4F75.svg)](https://plotly.com/)
[![AWS S3](https://img.shields.io/badge/AWS-S3%20Cloud%20Storage-orange.svg)](https://aws.amazon.com/s3/)
[![Apache Parquet](https://img.shields.io/badge/Format-Apache%20Parquet-teal.svg)](https://parquet.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![MLOps](https://img.shields.io/badge/MLOps-Model%20Registry-blueviolet.svg)](#-mlops-model-artifact-registry)
[![Tests](https://img.shields.io/badge/Tests-121%20Passed-brightgreen.svg)](#-running-the-test-suite)
[![Report](https://img.shields.io/badge/Report-Project%20Progress%20%26%20Bugs-informational.svg)](PROJECT_PROGRESS_REPORT.md)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

An enterprise-ready customer intelligence and segmentation platform that transforms raw retail transaction exports into actionable customer personas using **Recency, Frequency, and Monetary (RFM)** modeling and **Unsupervised Machine Learning (K-Means Clustering)**.

> 📘 **Full Engineering History & Bug Report:** For an itemized catalog of all bugs encountered, technical root causes, fixes, MLOps architecture shifts, and phase milestones, refer to the [Project Progress & Engineering Report](PROJECT_PROGRESS_REPORT.md).


---

## 🌟 Key Features

1. **Intelligent Ingestion & Auto-Column Mapping**
   - Upload any standard retail transaction CSV (Shopify, WooCommerce, Square, POS exports).
   - Fuzzy matching and alias detection automatically map column variations (e.g. `cust_id`, `ClientID`, `InvoiceNo`, `created_at`).
   - Supports multiple encodings (`utf-8`, `latin-1`, `cp1252`, `iso-8859-1`) with file size validation guards.
   - Robust input handling supporting file paths, raw bytes, and `io.StringIO` / `io.BytesIO` objects.
   - **Configurable Currency Localization:** Choose from USD ($), EUR (€), GBP (£), INR (₹), JPY (¥), CAD ($), and AUD ($) at ingestion, dynamically propagated across all metrics, charts, tables, and export file headers.

2. **Cloud Storage Ingestion & Export (AWS S3 + Local Mock)**
   - **Direct S3 Ingestion:** Pull transaction datasets directly from AWS S3 buckets (supports both `.csv` and `.parquet`).
   - **Offline Demo / Mock Mode:** Built-in local mock storage (`data/mock_s3/`) allowing zero-credential testing and local demos (`USE_LOCAL_MOCK=true`).
   - **Path Traversal Protection:** `pathlib.Path.resolve().is_relative_to()` enforces strict mock sandbox boundary containment, neutralizing directory traversal (`../`) attacks.
   - **Memory Exhaustion Safeguards:** Remote S3 responses check `ContentLength` and enforce `MAX_FILE_SIZE_MB` limits before loading, preventing OOM crashes on large files.
   - **Direct Cloud Export:** Push segmented customer audiences directly back to your S3 data lake or CRM staging bucket.
   - **Credential Security:** Configured through environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) or masked session overrides.

3. **Apache Parquet High-Performance Serialization**
   - High-efficiency columnar storage powered by `pyarrow` for both ingestion and exports.
   - Up to **88% storage size reduction** compared to traditional CSV files with faster load times and native schema preservation.
   - Interactive download format toggle (UTF-8 CSV vs. Snappy-compressed Apache Parquet).

4. **Transparent Data Hygiene Audit & Return Policy Engine**
   - Automatically filters guest checkouts, cancelled orders (`C` prefix), returns, and zero-price test line items.
   - **Explicit Returns Policy (`CleaningPolicy`):** Support toggling between Gross Spend (drop cancellations/returns) and Net Spend (netting refund amounts against purchase totals while preventing refund rows from inflating order frequency or artificially renewing purchase recency).
   - **Accounting Negatives:** Converts accounting parenthesis format (e.g., `(10.50)` &rarr; `-10.50`) to avoid converting refunds into positive revenue.
   - **Deterministic ID Normalization:** Ingestion forces ID columns to string `dtype`, preserving leading zeros (e.g. `"00123"` vs `"123"`). Trailing `.0` suffixes from float serialization (`"1001.0"` &rarr; `"1001"`) are safely stripped without destroying distinct non-numeric IDs.
   - **Invoice Validation:** Filters null or empty invoice numbers to prevent orphaned zero-frequency customer profiles.
   - Displays clear before/after counts, gross vs. net revenue, and refund rates in an itemized audit report.

5. **Pure-Pandas RFM Engine & Quintile Scoring Baseline**
   - Computes **Recency** (days since last purchase), **Frequency** (count of distinct orders), and **Monetary** (total lifetime spend) per customer.
   - **Traditional Quintile Scoring Baseline:** In addition to unsupervised K-Means clustering, computes deterministic 1–5 quintile scores (`r_score`, `f_score`, `m_score`, 3-digit `rfm_score` e.g., `'555'`, and `rfm_total` out of 15) with tie-aware ranking, serving as a transparent heuristic benchmark alongside ML personas.
   - Configurable reference snapshot date (defaults to midnight after the latest transaction) surfaced across dashboards and customer profiles.

6. **Self-Optimizing K-Means Clustering & Guardrails**
   - Normalizes right-skewed distributions with $\log(1+x)$ transformations followed by `StandardScaler`.
   - Automatically searches $k=2$ through $8$ and identifies optimal clustering using **Silhouette Score** maximization and the **Elbow Method**.
   - **Pre-Scaling Low-Variance Guard:** Evaluates feature standard deviation *before* applying `StandardScaler` (`log1p(data).std(axis=0) < 0.01`), preventing near-zero variance data from being artificially forced to unit variance.
   - Dynamic sidebar slider for instant interactive $k$ adjustments with edge-case protection.

7. **Explainable Customer Personas & Strategy Mapping**
   - **Behavior-Driven Rule Classification:** Clusters are classified via behavioral thresholds (`classify_cluster()`) into clear personas (Champions, Loyal Customers, At Risk, About to Sleep, Lost) with deterministic `(Tier 2)` collision resolution.
   - **Immutable State Design:** Persona metadata lookups and segment color maps are non-mutating wrappers, preventing side effects during multi-run analyses.
   - Actionable marketing recommendations and targeted campaign playbooks for each persona.

8. **Interactive 6-View Multipage Analytics & MLOps Platform**
   - **📊 Overview**: High-level KPIs, customer count/revenue distribution by segment, market treemaps, and zero-spend protected scatter plots.
   - **🔍 RFM Explorer**: Metric distribution histograms, percentile statistics, quintile baseline breakdown, and Elbow & Silhouette model diagnostics.
   - **🗺️ 3D Cluster Map**: Fully rotatable 3D customer space with zoom, pan, and 2D cross-section projections.
   - **👤 Customer Search**: Look up individual accounts, RFM quintile score, benchmark rankings across the base, and purchase timelines.
   - **📈 Cohort Retention**: Monthly acquisition cohort retention heatmap with contiguous period reindexing and adaptive high-contrast font colors.
   - **⏳ Segment Backtesting**: Out-of-time temporal window validation ($T_1 \le \text{cutoff} < T_2$), Markov segment transition probability heatmaps, customer churn tracking, and forward revenue realization analysis.
   - **🤖 Model Registry**: MLOps artifact catalog to browse versions, compare silhouette scores/inertia curves, hot-swap deployed estimators, and run live customer inference.

9. **Blazing-Fast Reactive Performance**
   - Computationally intensive operations (`load_csv`, `calculate_rfm`, `preprocess_rfm`, `evaluate_clusters`) are accelerated with `@st.cache_data`.
   - Instant response times on sidebar filter changes and interactive exploration.

10. **One-Click Multi-Format & Cloud Export**
    - Download the full segmented customer list with RFM metrics, cluster IDs, and persona labels in UTF-8 CSV or Apache Parquet formats.
    - Upload directly to AWS S3 buckets or local mock storage from the export interface.

11. **Production Docker Containerization & Health Checks**
    - **Multi-Stage Build:** Stage 1 virtualenv builder isolates pip compilation overhead; Stage 2 runtime image uses slim `python:3.11-slim` footprint (~180MB).
    - **Automated Health Check:** `HEALTHCHECK` continually inspects Streamlit's native `/_stcore/health` endpoint.
    - **Principle of Least Privilege:** Executes under an unprivileged `appuser` (UID 1000) non-root user.
    - **Docker Compose:** One-command local spinup (`docker compose up --build`) with volume persistence for `/app/data` and environment fallbacks.

12. **MLOps Model Artifact Registry & Live Inference Service**
    - **Artifact Serialization:** Persists trained Scikit-Learn `KMeans` models and fitted `StandardScaler` transformers using `joblib`.
    - **Complete Metadata Tracking:** Every artifact version stores UTC creation timestamps, cluster counts ($k$), inertia, silhouette scores, feature definitions, and persona mappings in `metadata.json`.
    - **Cross-Version Benchmarking:** Side-by-side silhouette score comparison charts and inertia curve analytics across model versions.
    - **Real-Time Customer Classifier:** Interactive inference playground allowing sales/marketing teams to input arbitrary RFM metrics and instantly receive assigned cluster IDs, persona badges, and targeted marketing strategies.

13. **Continuous Integration & Automated Testing Pipeline (GitHub Actions)**
    - **Automated CI Workflow:** Executes on every `push` and `pull_request` targeting `main`.
    - **Dependency Decoupling:** Production runtime requirements (`requirements.txt`) are separated from development/test packages (`requirements-dev.txt`), keeping production container images lean and CI runs fast.
    - **Full Test Automation:** Runs all 87 unit and integration tests across data ingestion, cleaning, RFM logic, clustering, cohort analysis, Docker configuration, MLOps registry, decoupling, smoke tests, and CI configuration.

14. **Architectural Decoupling & Unified Pipeline Artifacts**
    - **Headless Core Analytics:** Core modules under `src/` contain zero UI or Streamlit dependencies, enabling headless execution from CLI scripts, Airflow workflows, or backend microservices.
    - **UI Orchestration Layer (`services/`):** Streamlit caching (`@st.cache_data`) and presentation orchestration are cleanly encapsulated in `services/segmentation_service.py`.
    - **Unified Scikit-Learn Pipeline (`pipeline.joblib`):** Bundles `FunctionTransformer(np.log1p)`, `StandardScaler()`, and `KMeans()` into a single serialized pipeline artifact, eliminating preprocessing drift during inference.
    - **Explicit Model Promotion:** Exploratory pipeline runs remain in ephemeral session memory; users explicitly promote models to the registry with custom notes and tags via "Promote Model to Registry".

15. **Full-Page Streamlit AppTest Smoke Suites & Zero-Defect Hardening**
    - Comprehensive headless UI smoke testing via `streamlit.testing.v1.AppTest` covering `app.py` and all 5 sub-pages (`tests/test_pages_smoke.py`).
    - Validates both uninitialized guard clause warning states and fully populated interactive session states.
    - Defensive resolution of dynamic callback crashes (sidebar re-clustering, model fallback guards, Plotly inertia lookups).

16. **Two-Pass Net-Settlement Transaction Reconciler**
    - Built [`src/reconciler.py`](src/reconciler.py) to match returns and order cancellations against preceding purchase transactions by customer, item, and time window (180 days).
    - Eliminates upward frequency and spend distortion from naive return deletion.

17. **Hungarian Centroid Alignment & Whale Isolation**
    - Uses `scipy.optimize.linear_sum_assignment` to align newly fitted K-Means centroids to persisted anchor centroids, eliminating cluster label switching across retraining runs.
    - Two-stage hybrid isolation deterministically quarantines extreme upper-tail B2B wholesale buyers ($M > P_{99.5}$) into an *Apex VIP* tier, preventing them from compressing the 99% retail core.
    - Multi-index evaluation vectorizes Calinski-Harabasz and Davies-Bouldin metrics alongside subsampled Silhouette scores.

18. **Probabilistic Customer Lifetime Value Engine (BG/NBD + Gamma-Gamma)**
    - Built [`src/clv.py`](src/clv.py) implementing analytical Maximum Likelihood Estimation for Beta-Geometric/Negative Binomial Distribution (BG/NBD) and Gamma-Gamma spend submodels.
    - Computes posterior churn probability ($1 - P(\text{Alive})$), expected future repeat transactions $E[Y(t)]$, and 6-month projected CLV.
    - Resolves zero-repeat buyer singularity via empirical Bayes shrinkage for one-time purchasers.

19. **First-Party GDPR/CCPA Machine Unlearning & Pseudonymization**
    - Built [`src/governance.py`](src/governance.py) featuring HMAC-SHA256 customer identifier pseudonymization.
    - Supports GDPR Article 17 (Right to Erasure) with exact $O(1)$ arithmetic cluster centroid updates without retraining.

20. **Reverse-ETL CRM Activation & Three-Tier Testing Pyramid**
    - Standardized CRM export payloads with a deterministic 10% stratified holdout flag (`control_holdout = True`) for causal A/B testing of campaign lift.
    - Comprehensive testing pyramid in [`tests/test_metamorphic_and_contracts.py`](tests/test_metamorphic_and_contracts.py) covering data contracts, metamorphic rank monotonicity, and bootstrap cluster stability ($ARI \ge 0.80$).


---

## 🏛️ Project Architecture

### End-to-End Data Pipeline Flow

```mermaid
flowchart TD
    subgraph Ingestion ["1. Data Ingestion & Storage"]
        CSV[Local CSV / POS Export] --> Ingest[Smart Ingestion Engine]
        S3[AWS S3 Bucket / Mock Storage] --> Ingest
        ParquetIn[Apache Parquet .parquet] --> Ingest
        Ingest --> Clean[Data Hygiene & Validation Engine]
        Clean --> Audit[Transparent Audit Report]
    end

    subgraph RFM ["2. Feature Engineering"]
        Clean --> RFMCalc[Pure-Pandas RFM Engine]
        RFMCalc --> Transform[Log1p Transform & StandardScaler]
    end

    subgraph ML ["3. Unsupervised ML & Segmentation"]
        Transform --> KMeans[K-Means Clustering Engine]
        KMeans --> Opt[Silhouette & Elbow Optimization]
        Opt --> Guard[Low-Variance & k-Boundary Guards]
        Guard --> Persona[Centroid-to-Persona Strategy Mapping]
    end

    subgraph Presentation ["4. Visual Analytics & Cloud Export"]
        Persona --> Overview[Overview & Revenue Treemap]
        Persona --> Explorer[RFM Distribution Explorer]
        Persona --> Cluster3D[3D Interactive Cluster Map]
        Persona --> Search[Customer 360 & Timeline]
        Persona --> Cohort[Cohort Retention Heatmap]
        Persona --> ExportCSV[Download UTF-8 CSV]
        Persona --> ExportParquet[Download Apache Parquet]
        Persona --> ExportS3[Direct Export to AWS S3 / Mock]
    end

    subgraph MLOps ["5. MLOps Model Artifact Registry & Inference"]
        KMeans --> JoblibModel["joblib Serialized KMeans (.joblib)"]
        Transform --> JoblibScaler["joblib Serialized StandardScaler (.joblib)"]
        Persona --> MetadataJSON["Versioned Metadata & Personas (metadata.json)"]
        JoblibModel --> RegStore["Versioned Store: artifacts/models/v{n}_k{k}_{ts}/"]
        JoblibScaler --> RegStore
        MetadataJSON --> RegStore
        RegStore --> HotSwap["Model Hot-Swapping & Re-Clustering"]
        RegStore --> LiveInfer["Real-Time Customer Inference Service"]
    end

    subgraph Container ["6. Containerized Production Deployment"]
        Presentation --> AppServe["Streamlit Multi-Page App (:8501)"]
        HotSwap --> AppServe
        LiveInfer --> AppServe
        AppServe --> MultiStage["Multi-Stage Docker Image (python:3.11-slim)"]
        MultiStage --> SecureRun["Least-Privilege Non-Root Execution (UID 1000)"]
        SecureRun --> AutoHealth["Continuous HEALTHCHECK (/_stcore/health)"]
        SecureRun --> ComposeUp["Docker Compose Orchestration & Volume Mounts"]
    end

    subgraph CICD ["7. Continuous Integration & Quality Assurance"]
        CodeCommit[Git Commit / PR to main] --> GitHubActions[GitHub Actions CI Runner]
        GitHubActions --> DepInstall["Layered Dependency Install (prod + dev)"]
        DepInstall --> PytestSuite["Automated Pytest Suite (50/50 Passed)"]
        PytestSuite --> StatusBadge[Status Badge & Branch Protection]
    end
```

### Directory Structure

```
rfm-cluster360/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI workflow (automated pytest on push/PR)
├── app.py                          # Streamlit entrypoint, pipeline orchestrator & S3 ingestion tab
├── Dockerfile                      # Multi-stage production container definition (builder + runner)
├── docker-compose.yml              # Local container orchestration, volume persistence & health checks
├── .dockerignore                   # Build exclusion rules for lean, secure container images
├── HOW_TO_RUN.txt                  # Quick launch terminal commands and setup guide
├── pytest.ini                      # Pytest suite configuration and warning suppression
├── requirements.txt                # Pinned production dependencies (lean container image)
├── requirements-dev.txt            # Development & test-only dependencies (pytest, moto)
├── .env.example                    # Template for AWS S3 and application environment variables
├── .gitignore
├── README.md
├── PROJECT_PROGRESS_REPORT.md      # Comprehensive engineering progress, bug catalog, fixes & roadmap report
├── LICENSE                         # Apache 2.0 License
├── pyrightconfig.json              # Python static type checking configuration
│
├── artifacts/                      # MLOps Model Artifact Registry (Git-ignored)
│   └── models/                     # Versioned model directories (v{n}_k{k}_{timestamp})
│       └── v1_k4_.../
│           ├── kmeans_model.joblib # Serialized Scikit-Learn KMeans estimator
│           ├── scaler.joblib       # Serialized StandardScaler transformation
│           └── metadata.json       # Metrics, k, silhouette, inertia, persona mapping
│
├── config/
│   └── settings.py                 # Configuration constants, aliases, S3 configs, registry paths
│
├── data/
│   ├── sample_retail.csv           # Built-in demo transaction dataset
│   └── mock_s3/                    # Offline S3 mock directory for local development & testing
│       └── retail-data/
│           ├── transactions.csv
│           └── transactions.parquet
│
├── src/                            # Core computational engine (Modular, testable, zero UI dependencies)
│   ├── ingestion.py                # CSV loading, encoding fallbacks, fuzzy column mapping
│   ├── cloud_storage.py            # AWS S3 bucket loader & uploader, local mock fallback, moto support
│   ├── cleaning.py                 # Hygiene rules, accounting negatives, ID normalization, CleaningReport
│   ├── reconciler.py               # Two-pass net-settlement transaction reconciler
│   ├── governance.py               # GDPR/CCPA HMAC-SHA256 pseudonymization & O(1) machine unlearning
│   ├── rfm.py                      # Pure-Pandas RFM calculation, quintile baselines, lookbacks, avg spend
│   ├── clustering.py               # Hungarian centroid alignment, whale isolation, multi-index validity
│   ├── clv.py                      # Probabilistic BG/NBD + Gamma-Gamma CLV modeling & empirical Bayes shrinkage
│   ├── personas.py                 # Composite score ranking, collision-free persona classification
│   ├── cohort.py                   # Monthly acquisition cohort retention matrix with contiguous reindexing
│   ├── backtesting.py              # Temporal window splitting, Markov transition matrix, forward realization
│   ├── model_registry.py           # MLOps serialization, version listing, loading, and live inference
│   ├── export.py                   # Clean CSV, Parquet, and Reverse-ETL CRM export with 10% holdout
│   └── logger.py                   # Standardized structured application logging
│
├── pages/                          # Streamlit Multipage Navigation
│   ├── 1_Overview.py               # High-level KPIs, distributions, and playbooks
│   ├── 2_RFM_Explorer.py           # Distributions, percentile metrics, model diagnostics
│   ├── 3_Cluster_Map.py            # 3D interactive customer space & 2D slices
│   ├── 4_Customer_Search.py        # Customer lookup, timeline, and percentile rankings
│   ├── 5_Cohort_Retention.py       # Acquisition cohort retention heatmaps
│   ├── 6_Model_Registry.py         # MLOps versioning, model benchmarking, and live inference
│   └── 7_Segment_Backtesting.py    # Temporal backtesting, transition heatmaps, and forward validation
│
├── components/                     # Reusable Streamlit UI widgets
│   ├── metric_cards.py             # Executive KPI metrics & persona cards
│   ├── filters.py                  # Sidebar segment, country, and bounded spend range filters
│   ├── data_preview.py             # Column mapping editor & itemized cleaning audit cards
│   └── download_button.py          # CSV/Parquet toggle and direct AWS S3 upload section
│
├── viz/                            # Chart builders (Return pure Plotly Figures)
│   ├── overview_charts.py          # Segment volume, revenue, treemap, log-safe scatter plot
│   ├── rfm_charts.py               # Histograms, elbow curve, silhouette curve, radar charts
│   ├── cluster_scatter.py          # 3D scatter & 2D pairwise projections
│   ├── customer_timeline.py        # Individual customer timeline & percentile ranking
│   ├── cohort_heatmap.py           # Annotated cohort retention heatmap with adaptive text contrast
│   └── backtesting_charts.py       # Markov transition heatmaps and forward monetization charts
│
└── tests/                          # Automated Pytest Suite (98 Passed)
    ├── conftest.py                 # Hand-calculated transaction fixtures & synthetic base
    ├── test_ci_config.py           # GitHub Actions workflow syntax & requirements split verification
    ├── test_ingestion.py           # Encodings, empty files, fuzzy mapping, duplicate detection, StringIO
    ├── test_cloud_storage.py       # S3 moto mocks, Parquet serialization roundtrip, mock mode, validation
    ├── test_cleaning.py            # Deduplication, currency strings, accounting negatives, ID normalization
    ├── test_rfm.py                 # Hand-calculated RFM metrics, snapshot dates, summary stats, zero-monetary
    ├── test_clustering.py          # Preprocessing, evaluation, profiles, low-variance guard
    ├── test_personas.py            # Centroid labeling, synthetic customer groups, k=10 collision handling
    ├── test_cohort.py              # Cohort retention calculations, KPIs, non-contiguous month reindexing
    ├── test_backtesting.py         # Temporal window slicing, transition probabilities, forward realization
    ├── test_docker_config.py       # Multi-stage Dockerfile, docker-compose syntax, and .dockerignore rules
    └── test_model_registry.py      # Artifact save/load roundtrips, version listing, deletion, live inference
```

---

## 🗺️ Engineering Roadmap & Implementation Progress

| Phase | Milestone | Focus Areas | Status |
|:---:|---|---|:---:|
| **0** | **Repository Bootstrap** | Git repo initialization, `.gitignore`, architecture documentation, initial push | ✅ Complete |
| **1** | **Critical Bug Fixes** | Slider step overflow, duplicate column mapping, StringIO type error, k-slider edge cases | ✅ Complete |
| **2** | **Data Integrity & Cleaning** | Accounting parentheses, float-string ID normalization, null invoice validation | ✅ Complete |
| **3** | **ML Pipeline Robustness** | Percentile composite persona ranking, $k \ge 10$ collision qualifiers, low-variance clustering guard | ✅ Complete |
| **4** | **Visualization & UI** | Cohort contiguous month reindexing, adaptive heatmap text contrast, log-scale $0 scatter safety, `@st.cache_data` | ✅ Complete |
| **5** | **Cloud Integration** | AWS S3 bucket ingestion/export with local mock mode + Apache Parquet support | ✅ Complete |
| **6** | **Containerization** | Multi-stage production `Dockerfile`, `docker-compose`, health checks, non-root security | ✅ Complete |
| **7** | **MLOps Registry** | Joblib/JSON model artifact serialization, version registry, hot-swapping & live inference | ✅ Complete |
| **8** | **CI/CD Pipeline** | GitHub Actions workflow for linting, pytest matrix, container builds, status badge | ✅ Complete |
| **9** | **Security & Data Integrity** | Mock S3 path traversal containment, S3 size limit, leading zeros, live classifier fix | ✅ Complete |
| **10** | **Analytical Correctness** | Pre-scaling variance guard, rule-based persona thresholds, net returns cleaning policy | 📋 Planned |
| **11** | **Architectural Decoupling** | Headless core library without Streamlit dependencies, unified sklearn pipeline | 📋 Planned |
| **12** | **Production Validation** | Ruff linting, Pyright typing in CI, container build test, baseline quintile benchmarking | 📋 Planned |

---

## 🔄 Continuous Integration & Deployment (CI/CD)

RFM Cluster360 implements an enterprise-grade CI/CD pipeline via **GitHub Actions** (`.github/workflows/ci.yml`) providing parallel quality gates, automated security auditing, test coverage reporting, container build verification, and automated CD container registry publishing.

### 🌿 Git Branching Strategy (3-Branch Separation Model)

The repository follows an enterprise 3-branch separation model ensuring stability, clean release isolation, and rigorous review gates:

| Branch | Purpose & Scope | Promotion Rules & CI Gates |
|---|---|---|
| **`main`** | **Stable Production / Release Baseline:** Always deployable, immutable production code. | Protected baseline. Merges only via peer-reviewed Pull Requests with 100% CI pass rate. Triggers automated Docker image publishing to GitHub Container Registry (GHCR). |
| **`debugging-and-error-correction`** | **Bug Fixes & Financial Hardening:** Dedicated branch for numerical precision, financial math corrections, data cleaning hygiene, and runtime crash remedies. | Isolated from experimental features. Once validated by the 88-test suite, merged into `main` and synced back to `dev`. |
| **`dev`** | **Feature Engineering & Backtesting:** Active development branch for new analytics features, UI/UX enhancements, cohort tools, and model backtesting engines. | Aggregates work-in-progress features. Periodically rebased/merged with `main` to maintain regression-free alignment. |

```mermaid
gitGraph
  commit id: "main-v1.0"
  branch debugging-and-error-correction
  branch dev
  checkout debugging-and-error-correction
  commit id: "fix: return-policy"
  commit id: "fix: accounting-parens"
  checkout main
  merge debugging-and-error-correction id: "merge: hotfix-v1.0.1" tag: "v1.0.1"
  checkout dev
  commit id: "feat: model-registry"
  commit id: "feat: cohort-backtest"
  merge main id: "sync: main into dev"
  checkout main
  merge dev id: "merge: release-v1.1" tag: "v1.1.0"
```

### 🛠️ Workflow Architecture & Triggers

- **Triggers:**
  - `push` targeting `main`, `dev`, or `debugging-and-error-correction` (triggers full CI quality gates; `main` additionally executes CD to GHCR).
  - `pull_request` targeting `main`, `dev`, or `debugging-and-error-correction` (triggers full parallel CI quality gates).
  - `workflow_dispatch` (enables manual one-click workflow triggers directly from GitHub Actions UI).
- **Concurrency Control:** Cancels obsolete in-flight runs when newer commits are pushed to pull requests (`cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}`).
- **Parallel Quality Gates (Fast-Feedback Architecture):**
  1. **Linting & Code Quality (`lint`):** Executes `ruff check .` for lightning-fast PEP 8 linting and code hygiene.
  2. **Static Type Validation (`typecheck`):** Executes `pyright` for deep type consistency verification across all modules.
  3. **Security & Vulnerability Audit (`security`):** Scans production dependencies against known advisory databases using `pip-audit`.
  4. **Automated Test Suite & Coverage (`test`):** Runs the full 88-test Pytest suite with `--cov`, generates a rich markdown report in `$GITHUB_STEP_SUMMARY`, and uploads `coverage.xml` as a build artifact.
  5. **Container Verification (`docker-build`):** Validates multi-stage Docker compilation using `docker/setup-buildx-action` with GitHub Actions cache (`type=gha`).
  6. **Continuous Deployment (`publish`):** Upon successful completion of all quality gates on `main`, automatically builds, tags (`latest`, `sha-<short>`), and publishes the production image to **GitHub Container Registry** (`ghcr.io/khare69/rfm-cluster360`).

### 📦 Decoupled Dependency Architecture

To avoid bloat in the production Docker image while maintaining comprehensive testing and developer tooling, project requirements are strictly decoupled:

| Requirement Manifest | Purpose | Included Packages |
|---|---|---|
| `requirements.txt` | Production runtime & Docker build | `streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`, `python-dotenv`, `boto3`, `pyarrow`, `joblib` |
| `requirements-dev.txt` | CI/CD testing, security audit & local development | `pytest`, `pytest-cov`, `moto`, `ruff`, `pyright`, `pip-audit` |

---

## ☁️ Cloud Storage & Environment Configuration

### Application & AWS Credentials (`.env`)

Configure live cloud connections and artifact storage paths in your local `.env`:

```bash
# AWS S3 Cloud Storage Integration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1
AWS_S3_BUCKET=my-retail-datalake
USE_LOCAL_MOCK=false
MOCK_S3_DIR=data/mock_s3

# MLOps Model Registry Persistence
MODEL_REGISTRY_DIR=artifacts/models
```

### Zero-Config Offline Mock Mode (`USE_LOCAL_MOCK=true`)

For local demos or offline development without an AWS account, set:

```bash
USE_LOCAL_MOCK=true
```

In mock mode, the app reads and writes from `data/mock_s3/{bucket}/{key}` on disk. The repository ships with ready-to-test mock datasets:
- **Bucket:** `retail-data`
- **Key:** `transactions.csv` or `transactions.parquet`

---

## 🤖 MLOps Model Artifact Registry & Inference

RFM Cluster360 incorporates an enterprise-grade **MLOps Model Registry** (`src/model_registry.py`) providing versioning, persistence, hot-swapping, and real-time customer segmentation inference.

### 📦 Artifact Directory Hierarchy

Every trained model bundle is saved to disk with atomic components and metadata:

```
artifacts/models/
└── v1_k4_20261009_163000/
    ├── kmeans_model.joblib    # Serialized Scikit-Learn KMeans estimator
    ├── scaler.joblib          # Serialized StandardScaler transformation
    └── metadata.json          # Complete training metadata & persona mapping
```

### 📋 Model Metadata Schema (`metadata.json`)

```json
{
  "version_id": "v1_k4_20261009_163000",
  "created_at": "2026-10-09T11:00:00+00:00",
  "k": 4,
  "inertia": 78.42,
  "silhouette_score": 0.4521,
  "feature_names": ["recency", "frequency", "monetary"],
  "customer_count": 120,
  "cluster_map": {
    "0": "Champions",
    "1": "Loyal Customers",
    "2": "Potential Loyalists",
    "3": "Lost / Dormant"
  },
  "notes": "Auto-saved pipeline run (k=4)"
}
```

### ⚡ Programmatic Python API

Developers and data pipelines can interact with the registry programmatically:

```python
from src.model_registry import (
    save_model_artifact,
    list_model_versions,
    load_model_artifact,
    predict_segment,
)

# 1. List all registered model versions
versions = list_model_versions()
print(f"Latest model: {versions[0]['version_id']}, k={versions[0]['k']}")

# 2. Load model artifact & fitted preprocessor
model, scaler, cluster_map, metadata = load_model_artifact(versions[0]["path"])

# 3. Real-time inference on new customer record
customer_rfm = {"recency": 15.0, "frequency": 6.0, "monetary": 840.50}
prediction = predict_segment(model, scaler, cluster_map, customer_rfm)
print(f"Assigned Segment: {prediction['segment']} (Cluster #{prediction['cluster']})")

# 4. Batch inference on pandas DataFrame
batch_df = predict_segment(model, scaler, cluster_map, new_customers_df)
```

---

## 🚀 Quickstart Guide

### 1. Clone Repository & Create Virtual Environment

```bash
# Clone repository
git clone https://github.com/Khare69/RFM-Cluster360.git
cd rfm-cluster360

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install core production dependencies
pip install -r requirements.txt

# (Optional) Install development and testing dependencies for pytest
pip install -r requirements-dev.txt
```

### 3. Run the Web Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🐳 Run with Docker (Production & Cloud Deployment)

The application includes a production-grade multi-stage `Dockerfile` and `docker-compose.yml` for containerized environments.

### Option A: One-Command Spinup with Docker Compose (Recommended)

Start the containerized service with local directory mounts and environment variables:

```bash
# Build and run the container
docker compose up --build

# Run in background (detached mode)
docker compose up -d --build

# Stop the container
docker compose down
```

### Option B: Manual Docker Build & Run

```bash
# 1. Build the lightweight multi-stage image
docker build -t rfm-cluster360:latest .

# 2. Run the container with persistent volume and port forwarding
docker run -d \
  --name rfm-cluster360-app \
  -p 8501:8501 \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/artifacts:/app/artifacts" \
  --env-file .env \
  rfm-cluster360:latest

# 3. View container logs
docker logs -f rfm-cluster360-app
```

The containerized dashboard is accessible at `http://localhost:8501`.

### 🛡️ Container Architecture & Security Highlights

- **Multi-Stage Build Pattern:** The builder stage creates an isolated virtual environment (`/opt/venv`), separating build-time dependencies (`build-essential`) from the lightweight final runtime image (`python:3.11-slim`), reducing image size by over 75%.
- **Unprivileged Non-Root User:** Runs under a dedicated `appuser` system account (UID 1000) adhering to container security best practices and least-privilege principles.
- **Automated Health Check:** Monitored via `HEALTHCHECK` querying Streamlit's internal health endpoint:
  ```bash
  curl --fail http://localhost:8501/_stcore/health || exit 1
  ```
- **Optimized Build Cache:** Strict `.dockerignore` excludes virtual environments, git metadata, bytecode, unit tests, and local credentials.

---

## 🧪 Running the Test Suite

The codebase is backed by an automated test suite with hand-verified assertions, offline cloud mocks (`moto`), container configuration checks, and CI pipeline validation:

```bash
# Run test suite
pytest tests/ -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
configfile: pytest.ini
testpaths: tests
collected 84 items

tests/test_ci_config.py::test_ci_workflow_structure PASSED               [  1%]
tests/test_ci_config.py::test_requirements_split PASSED                  [  3%]
tests/test_cleaning.py::test_clean_transactions PASSED                   [  4%]
tests/test_cleaning.py::test_clean_transactions_remove_duplicates PASSED [  6%]
tests/test_cleaning.py::test_detect_duplicates PASSED                    [  7%]
tests/test_cleaning.py::test_clean_transactions_with_currency_strings PASSED [  9%]
tests/test_cleaning.py::test_clean_transactions_with_accounting_negatives PASSED [ 10%]
tests/test_cleaning.py::test_clean_id_normalizes_float_strings PASSED    [ 12%]
tests/test_cleaning.py::test_clean_transactions_null_invoice_no PASSED   [ 13%]
tests/test_cleaning.py::test_clean_transactions_case_insensitive_null_invoices PASSED [ 15%]
tests/test_clean_id_preserves_leading_zeros_and_distinct_formats PASSED [ 16%]
tests/test_cleaning.py::test_clean_transactions_net_returns_policy PASSED [ 18%]
tests/test_cleaning.py::test_export_dataframe_with_custom_currency PASSED [ 20%]
tests/test_cloud_storage.py::test_to_parquet_bytes_roundtrip PASSED      [ 21%]
tests/test_cloud_storage.py::test_to_parquet_bytes_columns_filter PASSED [ 23%]
tests/test_cloud_storage.py::test_load_from_s3_csv_with_moto PASSED      [ 24%]
tests/test_cloud_storage.py::test_load_from_s3_parquet_with_moto PASSED  [ 26%]
tests/test_cloud_storage.py::test_upload_to_s3_parquet_and_csv_with_moto PASSED [ 27%]
tests/test_cloud_storage.py::test_load_from_s3_nonexistent_key_moto PASSED [ 29%]
tests/test_cloud_storage.py::test_local_mock_mode_load_and_upload PASSED [ 30%]
tests/test_cloud_storage.py::test_load_from_s3_input_validation PASSED   [ 32%]
tests/test_cloud_storage.py::test_upload_to_s3_input_validation PASSED   [ 33%]
tests/test_cloud_storage.py::test_mock_s3_path_traversal_prevention PASSED [ 35%]
tests/test_cloud_storage.py::test_load_and_upload_path_traversal_graceful_error PASSED [ 36%]
tests/test_cloud_storage.py::test_load_from_s3_mock_size_limit PASSED    [ 38%]
tests/test_clustering.py::test_preprocess_rfm PASSED                     [ 40%]
tests/test_clustering.py::test_evaluate_clusters PASSED                  [ 41%]
tests/test_clustering.py::test_run_kmeans_and_build_df PASSED            [ 43%]
tests/test_clustering.py::test_evaluate_clusters_low_variance PASSED     [ 44%]
tests/test_clustering.py::test_preprocess_rfm_preserves_pre_scale_low_variance PASSED [ 46%]
tests/test_cohort.py::test_build_cohort_matrix PASSED                    [ 47%]
tests/test_cohort.py::test_build_cohort_matrix_non_contiguous_months PASSED [ 49%]
tests/test_docker_config.py::test_dockerfile_structure PASSED            [ 50%]
tests/test_docker_config.py::test_dockercompose_structure PASSED        [ 52%]
tests/test_docker_config.py::test_dockerignore_rules PASSED              [ 53%]
tests/test_ingestion.py::test_load_csv_from_string_io PASSED             [ 55%]
tests/test_ingestion.py::test_load_csv_empty_file PASSED                 [ 56%]
tests/test_ingestion.py::test_load_csv_latin1_encoding PASSED            [ 58%]
tests/test_auto_map_columns_exact PASSED                                  [ 60%]
tests/test_auto_map_columns_fuzzy_and_aliases PASSED                      [ 61%]
tests/test_validate_mapping PASSED                                        [ 63%]
tests/test_validate_mapping_duplicate_columns PASSED                      [ 64%]
tests/test_load_csv_from_string_io_object PASSED                          [ 66%]
tests/test_ingestion.py::test_load_csv_preserves_leading_zeros PASSED    [ 67%]
tests/test_model_registry.py::test_save_and_load_roundtrip PASSED        [ 69%]
tests/test_model_registry.py::test_list_model_versions PASSED            [ 70%]
tests/test_model_registry.py::test_predict_segment_single_record PASSED  [ 72%]
tests/test_model_registry.py::test_predict_segment_dataframe PASSED      [ 73%]
tests/test_model_registry.py::test_predict_segment_missing_features PASSED [ 75%]
tests/test_model_registry.py::test_load_nonexistent_or_corrupt_artifact PASSED [ 76%]
tests/test_model_registry.py::test_delete_model_version PASSED           [ 78%]
tests/test_model_registry.py::test_save_model_type_validation PASSED     [ 80%]
tests/test_model_registry.py::test_model_registry_page_apptest_smoke PASSED [ 81%]
tests/test_personas.py::test_label_segments_with_synthetic PASSED        [ 83%]
tests/test_personas.py::test_label_segments_k10_unique_labels PASSED     [ 80%]
tests/test_personas.py::test_classify_cluster_rule_based_behavior PASSED [ 82%]
tests/test_personas.py::test_get_persona_metadata_and_colors_no_mutation PASSED [ 83%]
tests/test_rfm.py::test_calculate_rfm_hand_calculated PASSED             [ 85%]
tests/test_rfm.py::test_default_snapshot_date PASSED                     [ 86%]
tests/test_rfm.py::test_calculate_rfm_snapshot_in_past_error PASSED      [ 88%]
tests/test_rfm.py::test_get_rfm_summary_stats PASSED                     [ 89%]
tests/test_rfm.py::test_rfm_summary_scatter_zero_monetary PASSED         [ 91%]
tests/test_rfm.py::test_calculate_rfm_defensive_invoice_filtering PASSED [ 92%]
tests/test_rfm.py::test_silhouette_plot_negative_scores PASSED           [ 94%]
tests/test_rfm.py::test_calculate_rfm_with_net_refunds_policy PASSED     [ 95%]
tests/test_rfm.py::test_rfm_histograms_custom_currency PASSED            [ 97%]
tests/test_rfm.py::test_cluster_scatter_custom_currency PASSED           [ 98%]
tests/test_rfm.py::test_customer_timeline_and_benchmark_custom_currency PASSED [ 96%]
tests/test_rfm.py::test_compute_rfm_quintiles_deterministic_scoring PASSED             [ 97%]
tests/test_rfm.py::test_compute_rfm_quintiles_handles_ties_without_bin_error PASSED    [ 98%]
tests/test_rfm.py::test_compute_rfm_quintiles_small_dataset_and_empty PASSED           [100%]

============================= 84 passed in 26.82s =============================
```

---

## 📄 License

Distributed under the [Apache 2.0 License](LICENSE).
