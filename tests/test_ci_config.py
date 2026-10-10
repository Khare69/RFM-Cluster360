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
    assert "pytest" in content, "Workflow must execute pytest suite"


def test_requirements_split():
    """Verify development and test dependencies are properly separated from production dependencies."""
    prod_req_path = Path("requirements.txt")
    dev_req_path = Path("requirements-dev.txt")

    assert prod_req_path.exists(), "requirements.txt must exist at root"
    assert dev_req_path.exists(), "requirements-dev.txt must exist at root"

    prod_content = prod_req_path.read_text(encoding="utf-8")
    dev_content = dev_req_path.read_text(encoding="utf-8")

    # Test dependencies must NOT be in production requirements
    assert "pytest" not in prod_content, "pytest must not be listed in production requirements.txt"
    assert "moto" not in prod_content, "moto must not be listed in production requirements.txt"

    # Test dependencies MUST be in requirements-dev.txt
    assert "pytest" in dev_content, "pytest must be listed in requirements-dev.txt"
    assert "moto" in dev_content, "moto must be listed in requirements-dev.txt"

    # Core production packages must be present in requirements.txt
    core_packages = ["pandas", "numpy", "scikit-learn", "plotly", "streamlit", "boto3", "pyarrow", "joblib"]
    for pkg in core_packages:
        assert pkg in prod_content, f"Core production package '{pkg}' must be in requirements.txt"
