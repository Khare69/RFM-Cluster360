"""
Unit tests validating Docker configuration, multi-stage Dockerfile syntax, and compose definitions.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_structure():
    """Verify Dockerfile exists and enforces multi-stage security and healthcheck best practices."""
    dockerfile_path = REPO_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile must exist at repository root"

    content = dockerfile_path.read_text(encoding="utf-8")

    # Multi-stage build checks
    assert "AS builder" in content, "Dockerfile must define a builder stage"
    assert "AS runner" in content, "Dockerfile must define a runner stage"

    base_images = re.findall(r"FROM\s+python:(\d+)\.(\d+)-slim\b", content)
    assert len(base_images) >= 2, (
        "Dockerfile must use lightweight python:<major>.<minor>-slim base in builder and runner stages"
    )
    assert len(set(base_images)) == 1, (
        f"Dockerfile stages must use the same Python slim base version, found: {base_images}"
    )
    major, minor = (int(base_images[0][0]), int(base_images[0][1]))
    assert (major, minor) >= (3, 12), (
        f"Dockerfile Python base {major}.{minor} must be >= 3.12 for numpy>=2.5.4 compatibility"
    )

    # Security & Port checks
    assert "EXPOSE 8501" in content, "Dockerfile must expose port 8501"
    assert "appuser" in content, "Dockerfile must define a non-root application user"
    assert "USER appuser" in content, "Dockerfile must switch to non-root user for execution"

    # Healthcheck verification
    assert "HEALTHCHECK" in content, "Dockerfile must declare a HEALTHCHECK instruction"
    assert "/_stcore/health" in content, "HEALTHCHECK must target Streamlit health endpoint"


def test_docker_compose_structure():
    """Verify docker-compose.yml exists, defines app service, port mappings, and volume mounts."""
    compose_path = REPO_ROOT / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist at repository root"

    content = compose_path.read_text(encoding="utf-8")

    assert "services:" in content
    assert "app:" in content
    assert "8501:8501" in content, "docker-compose.yml must map port 8501"
    assert "./data:/app/data" in content, "docker-compose.yml must mount data directory volume"
    assert "_stcore/health" in content, "docker-compose.yml must configure service healthcheck"


def test_dockerignore_rules():
    """Verify .dockerignore excludes sensitive and unnecessary local artifacts."""
    dockerignore_path = REPO_ROOT / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore must exist at repository root"

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    expected_exclusions = [".git/", ".venv/", "__pycache__/", ".env", "tests/", ".pytest_cache/"]
    for pattern in expected_exclusions:
        assert any(pattern in line for line in lines), f"{pattern} must be excluded in .dockerignore"
