"""
Unit tests validating GitHub Actions CI/CD workflow configuration and requirements split.
"""

from pathlib import Path


def test_ci_workflow_structure():
    """Verify .github/workflows/ci.yml exists and enforces CI best practices."""
    workflow_path = Path(".github/workflows/ci.yml")
    assert workflow_path.exists(), "GitHub Actions workflow .github/workflows/ci.yml must exist"

    content = workflow_path.read_text(encoding="utf-8")

    # Workflow triggers
    assert "push:" in content, "Workflow must trigger on push"
    assert "pull_request:" in content, "Workflow must trigger on pull_request"
    assert "main" in content, "Workflow must target the main branch"

    # Runner & steps
    assert "ubuntu-latest" in content, "Workflow runner must be ubuntu-latest"
    assert "actions/checkout" in content, "Workflow must checkout repository code"
    assert "actions/setup-python" in content, "Workflow must set up Python environment"
    assert "3.11" in content, "Workflow must configure Python 3.11"
    assert "requirements.txt" in content, "Workflow must install production requirements"
    assert "requirements-dev.txt" in content, "Workflow must install development requirements"
    assert "ruff check" in content, "Workflow must execute ruff linter"
    assert "pyright" in content, "Workflow must execute pyright static type checker"
    assert "pytest" in content, "Workflow must execute pytest suite"
    assert "docker build" in content, "Workflow must verify Docker build"


def test_requirements_split():
    """Verify development and test dependencies are properly separated from production dependencies."""
    prod_req_path = Path("requirements.txt")
    dev_req_path = Path("requirements-dev.txt")

    assert prod_req_path.exists(), "requirements.txt must exist at root"
    assert dev_req_path.exists(), "requirements-dev.txt must exist at root"

    prod_content = prod_req_path.read_text(encoding="utf-8")
    dev_content = dev_req_path.read_text(encoding="utf-8")

    # Test & dev quality dependencies must NOT be in production requirements
    for dev_tool in ["pytest", "moto", "ruff", "pyright"]:
        assert dev_tool not in prod_content, f"{dev_tool} must not be listed in production requirements.txt"
        assert dev_tool in dev_content, f"{dev_tool} must be listed in requirements-dev.txt"

    # Core production packages must be present in requirements.txt
    core_packages = ["pandas", "numpy", "scikit-learn", "plotly", "streamlit", "boto3", "pyarrow", "joblib"]
    for pkg in core_packages:
        assert pkg in prod_content, f"Core production package '{pkg}' must be in requirements.txt"
