"""
Unit tests validating Docker configuration, multi-stage Dockerfile syntax, and compose definitions.
"""

from pathlib import Path


def test_dockerfile_structure():
    """Verify Dockerfile exists and enforces multi-stage security and healthcheck best practices."""
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists(), "Dockerfile must exist at repository root"

    content = dockerfile_path.read_text(encoding="utf-8")

    # Multi-stage build checks
    assert "AS builder" in content, "Dockerfile must define a builder stage"
    assert "AS runner" in content, "Dockerfile must define a runner stage"
    assert "python:3.11-slim" in content, "Dockerfile must use lightweight python:3.11-slim base"

    # Security & Port checks
    assert "EXPOSE 8501" in content, "Dockerfile must expose port 8501"
    assert "appuser" in content, "Dockerfile must define a non-root application user"
    assert "USER appuser" in content, "Dockerfile must switch to non-root user for execution"

    # Healthcheck verification
    assert "HEALTHCHECK" in content, "Dockerfile must declare a HEALTHCHECK instruction"
    assert "/_stcore/health" in content, "HEALTHCHECK must target Streamlit health endpoint"


def test_docker_compose_structure():
    """Verify docker-compose.yml exists, defines app service, port mappings, and volume mounts."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml must exist at repository root"

    content = compose_path.read_text(encoding="utf-8")

    assert "services:" in content
    assert "app:" in content
    assert "8501:8501" in content, "docker-compose.yml must map port 8501"
    assert "./data:/app/data" in content, "docker-compose.yml must mount data directory volume"
    assert "_stcore/health" in content, "docker-compose.yml must configure service healthcheck"


def test_dockerignore_rules():
    """Verify .dockerignore excludes sensitive and unnecessary local artifacts."""
    dockerignore_path = Path(".dockerignore")
    assert dockerignore_path.exists(), ".dockerignore must exist at repository root"

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    expected_exclusions = [".git/", ".venv/", "__pycache__/", ".env", "tests/", ".pytest_cache/"]
    for pattern in expected_exclusions:
        assert any(pattern in line for line in lines), f"{pattern} must be excluded in .dockerignore"
