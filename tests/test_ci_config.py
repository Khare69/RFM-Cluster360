"""
Unit tests validating GitHub Actions CI/CD workflow configuration and requirements split.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_ci_workflow_structure():
    """Verify .github/workflows/ci.yml exists and enforces CI best practices."""
    workflow_path = REPO_ROOT / ".github/workflows/ci.yml"
    assert workflow_path.exists(), "GitHub Actions workflow .github/workflows/ci.yml must exist"

    content = workflow_path.read_text(encoding="utf-8")

    # Workflow triggers
    assert "push:" in content, "Workflow must trigger on push"
    assert "pull_request:" in content, "Workflow must trigger on pull_request"
    assert "workflow_dispatch:" in content, "Workflow must support manual trigger workflow_dispatch"
    assert "main" in content, "Workflow must target the main branch"
    assert "dev" in content, "Workflow must target the dev branch"
    assert "debugging-and-error-correction" in content, "Workflow must target the debugging-and-error-correction branch"

    # Runner & steps
    assert "ubuntu-latest" in content, "Workflow runner must be ubuntu-latest"
    assert "actions/checkout" in content, "Workflow must checkout repository code"
    assert "actions/setup-python" in content, "Workflow must set up Python environment"
    assert "3.11" in content, "Workflow must configure Python 3.11"
    assert "requirements.txt" in content, "Workflow must install production requirements"
    assert "requirements-dev.txt" in content, "Workflow must install development requirements"
    assert "ruff check" in content, "Workflow must execute ruff linter"
    assert "pyright" in content, "Workflow must execute pyright static type checker"
    assert "pip-audit" in content, "Workflow must execute security vulnerability audit"
    assert "pytest" in content, "Workflow must execute pytest suite"
    assert "--cov=" in content, "Workflow must compute test coverage"
    assert "GITHUB_STEP_SUMMARY" in content, "Workflow must publish summary to GitHub Step Summary"
    assert "docker build" in content, "Workflow must verify Docker build"


def test_ci_cd_publishing_and_security_configuration():
    """Verify CD deployment, 3-branch workflow isolation, and container registry publishing pipeline."""
    workflow_path = REPO_ROOT / ".github/workflows/ci.yml"
    content = workflow_path.read_text(encoding="utf-8")

    # 3-Branch Separation Architecture
    assert '"main", "dev", "debugging-and-error-correction"' in content, (
        "Workflow must explicitly target the 3-branch separation model"
    )
    # CD deployment is strictly gated to the stable main release branch
    assert "github.ref == 'refs/heads/main'" in content, (
        "Automated CD container publishing must be strictly gated to the main production branch"
    )

    # GHCR Container Registry integration
    assert "ghcr.io" in content, "Workflow must configure GitHub Container Registry (ghcr.io)"
    assert "docker/login-action" in content, "Workflow must authenticate to registry"
    assert "docker/build-push-action" in content, "Workflow must use build-push-action"
    assert "packages: write" in content, "Workflow must have write permissions for packages"


def test_requirements_split():
    """Verify development and test dependencies are properly separated from production dependencies."""
    prod_req_path = REPO_ROOT / "requirements.txt"
    dev_req_path = REPO_ROOT / "requirements-dev.txt"

    assert prod_req_path.exists(), "requirements.txt must exist at root"
    assert dev_req_path.exists(), "requirements-dev.txt must exist at root"

    prod_content = prod_req_path.read_text(encoding="utf-8")
    dev_content = dev_req_path.read_text(encoding="utf-8")

    # Test & dev quality dependencies must NOT be in production requirements
    for dev_tool in ["pytest", "moto", "ruff", "pyright", "pytest-cov", "pip-audit", "pre-commit"]:
        assert dev_tool not in prod_content, f"{dev_tool} must not be listed in production requirements.txt"
        assert dev_tool in dev_content, f"{dev_tool} must be listed in requirements-dev.txt"

    # Core production packages must be present in requirements.txt
    core_packages = ["pandas", "numpy", "scikit-learn", "plotly", "streamlit", "boto3", "pyarrow", "joblib"]
    for pkg in core_packages:
        assert pkg in prod_content, f"Core production package '{pkg}' must be in requirements.txt"


def test_repository_governance_and_templates():
    """Verify Pull Request template, Issue templates, and Dependabot governance configuration."""
    pr_template_path = REPO_ROOT / ".github/PULL_REQUEST_TEMPLATE.md"
    bug_issue_path = REPO_ROOT / ".github/ISSUE_TEMPLATE/bug_report.yml"
    feat_issue_path = REPO_ROOT / ".github/ISSUE_TEMPLATE/feature_request.yml"
    dependabot_path = REPO_ROOT / ".github/dependabot.yml"

    # PR Template checks
    assert pr_template_path.exists(), "PULL_REQUEST_TEMPLATE.md must exist in .github/"
    pr_content = pr_template_path.read_text(encoding="utf-8")
    assert "`main`" in pr_content, "PR template must include main branch option"
    assert "`debugging-and-error-correction`" in pr_content, "PR template must include debugging branch option"
    assert "`dev`" in pr_content, "PR template must include dev branch option"
    assert "88+ automated tests" in pr_content, "PR template must require test suite pass"

    # Issue Templates checks
    assert bug_issue_path.exists(), "bug_report.yml must exist in .github/ISSUE_TEMPLATE/"
    bug_content = bug_issue_path.read_text(encoding="utf-8")
    assert "debugging-and-error-correction" in bug_content, "Bug report must target debugging branch"

    assert feat_issue_path.exists(), "feature_request.yml must exist in .github/ISSUE_TEMPLATE/"
    feat_content = feat_issue_path.read_text(encoding="utf-8")
    assert "dev" in feat_content, "Feature request must target dev branch"

    # Dependabot checks
    assert dependabot_path.exists(), "dependabot.yml must exist in .github/"
    dep_content = dependabot_path.read_text(encoding="utf-8")
    assert 'package-ecosystem: "pip"' in dep_content, "Dependabot must monitor pip dependencies"
    assert 'package-ecosystem: "github-actions"' in dep_content, "Dependabot must monitor GitHub Actions"


def test_developer_tooling_and_task_runners():
    """Verify pre-commit hooks configuration, Makefile targets, and PowerShell task runner."""
    pre_commit_path = REPO_ROOT / ".pre-commit-config.yaml"
    makefile_path = REPO_ROOT / "Makefile"
    ps_runner_path = REPO_ROOT / "scripts/run.ps1"

    # Pre-commit configuration verification
    assert pre_commit_path.exists(), ".pre-commit-config.yaml must exist at repository root"
    pre_content = pre_commit_path.read_text(encoding="utf-8")
    assert "trailing-whitespace" in pre_content, "Pre-commit must configure whitespace checking"
    assert "check-yaml" in pre_content, "Pre-commit must configure YAML validation"
    assert "ruff" in pre_content, "Pre-commit must configure Ruff hooks"

    # Makefile verification
    assert makefile_path.exists(), "Makefile must exist at repository root"
    make_content = makefile_path.read_text(encoding="utf-8")
    for target in ["test:", "lint:", "typecheck:", "run:", "docker-build:", "clean:"]:
        assert target in make_content, f"Makefile must define '{target}' target"

    # PowerShell task runner verification
    assert ps_runner_path.exists(), "scripts/run.ps1 must exist"
    ps_content = ps_runner_path.read_text(encoding="utf-8")
    for cmd in ["test", "lint", "typecheck", "run", "docker-build"]:
        assert cmd in ps_content, f"PowerShell runner must handle '{cmd}' command"


