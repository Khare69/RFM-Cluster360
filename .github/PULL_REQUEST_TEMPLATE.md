## 🌿 Pull Request Overview

### 1. Target Branch Selection
Please indicate which branch this Pull Request is targeting in accordance with the repository's 3-branch separation model:
- [ ] `main` — Stable production / release baseline (Requires full CI pass + code review).
- [ ] `debugging-and-error-correction` — Bug fixes, financial math corrections, and stability hardening.
- [ ] `dev` — Feature engineering, UI/UX upgrades, backtesting engines, and experimental analytics.

---

### 2. Description of Changes
*Provide a concise summary of the changes made, the problem being solved, and any architectural decisions.*

---

### 3. Classification of Changes
- [ ] 🐛 **Bug Fix:** Resolution of a runtime crash, UI edge case, or defensive validation defect.
- [ ] 💰 **Financial / Mathematical Correction:** Adjustment to net returns, accounting parens, or RFM quintile calculations.
- [ ] ✨ **New Feature:** Addition of new analytical capabilities, visualizations, or MLOps integrations.
- [ ] 📈 **Backtesting & Forecasting:** Historical simulation, Markov transition matrices, or survival analysis.
- [ ] 🛠️ **DevOps / CI/CD:** GitHub Actions workflows, Docker containerization, or dependencies.

---

### 4. Quality & Pre-Merge Checklist
Before requesting review or merging, verify that all automated gates have passed locally:
- [ ] **Pytest Test Suite:** All 88+ automated tests pass (`pytest -v`).
- [ ] **Linting & Code Style:** Ruff reports zero errors (`ruff check .`).
- [ ] **Static Type Validation:** Pyright reports zero errors (`pyright`).
- [ ] **Docker Compilation:** Container image builds without errors (`docker build -t rfm-cluster360:test .`).
- [ ] **Secrets & Security:** No API keys, AWS credentials, or sensitive files are staged.
- [ ] **Documentation:** `README.md` and `PROJECT_PROGRESS_REPORT.md` updated if relevant.
