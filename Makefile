# ==============================================================================
# Makefile for RFM Cluster360
# Multi-platform developer commands for testing, linting, and containerization
# ==============================================================================

.PHONY: help install test lint typecheck audit run docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  make install       Install production and development dependencies"
	@echo "  make test          Run automated Pytest suite with short traceback"
	@echo "  make lint          Run Ruff linter and code hygiene checks"
	@echo "  make typecheck     Run Pyright static type checker"
	@echo "  make audit         Run pip-audit security vulnerability scanner"
	@echo "  make run           Launch local Streamlit web application"
	@echo "  make docker-build  Compile multi-stage Docker container"
	@echo "  make docker-run    Launch container via Docker Compose"
	@echo "  make clean         Remove cache files and temporary artifacts"

install:
	pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest -v --tb=short

lint:
	ruff check .

typecheck:
	pyright

audit:
	pip-audit -r requirements.txt

run:
	streamlit run app.py

docker-build:
	docker build -t rfm-cluster360:test .

docker-run:
	docker compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
