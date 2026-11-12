<#
.SYNOPSIS
    RFM Cluster360 — Windows PowerShell Developer Task Runner

.DESCRIPTION
    Provides single-command shortcuts for running tests, linting, type-checking,
    application startup, and Docker builds on Windows systems.

.EXAMPLE
    .\scripts\run.ps1 test
    .\scripts\run.ps1 lint
    .\scripts\run.ps1 typecheck
    .\scripts\run.ps1 run
    .\scripts\run.ps1 docker
    .\scripts\run.ps1 clean
#>

param (
    [Parameter(Position=0, Mandatory=$false)]
    [ValidateSet("help", "install", "test", "lint", "typecheck", "audit", "run", "docker-build", "docker-run", "clean")]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"
$VenvPython = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$VenvPytest = if (Test-Path ".\.venv\Scripts\pytest.exe") { ".\.venv\Scripts\pytest.exe" } else { "pytest" }
$VenvRuff   = if (Test-Path ".\.venv\Scripts\ruff.exe")   { ".\.venv\Scripts\ruff.exe" }   else { "ruff" }
$VenvPyright = if (Test-Path ".\.venv\Scripts\pyright.exe") { ".\.venv\Scripts\pyright.exe" } else { "pyright" }
$VenvStreamlit = if (Test-Path ".\.venv\Scripts\streamlit.exe") { ".\.venv\Scripts\streamlit.exe" } else { "streamlit" }

switch ($Command) {
    "help" {
        Write-Host "=================================================================" -ForegroundColor Cyan
        Write-Host "              RFM Cluster360 PowerShell Task Runner              " -ForegroundColor Cyan
        Write-Host "=================================================================" -ForegroundColor Cyan
        Write-Host "Usage: .\scripts\run.ps1 <command>`n"
        Write-Host "Commands:"
        Write-Host "  install       Install production and development dependencies" -ForegroundColor Green
        Write-Host "  test          Run full Pytest automated test suite" -ForegroundColor Green
        Write-Host "  lint          Run Ruff linter and code style checks" -ForegroundColor Green
        Write-Host "  typecheck     Run Pyright static type checker" -ForegroundColor Green
        Write-Host "  audit         Run pip-audit security vulnerability scanner" -ForegroundColor Green
        Write-Host "  run           Launch Streamlit web application" -ForegroundColor Green
        Write-Host "  docker-build  Build multi-stage Docker image" -ForegroundColor Green
        Write-Host "  docker-run    Start application via Docker Compose" -ForegroundColor Green
        Write-Host "  clean         Remove cache and temporary directories" -ForegroundColor Green
    }

    "install" {
        Write-Host "==> Installing dependencies..." -ForegroundColor Yellow
        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install -r requirements.txt -r requirements-dev.txt
    }

    "test" {
        Write-Host "==> Executing Pytest test suite..." -ForegroundColor Yellow
        & $VenvPytest -v --tb=short
    }

    "lint" {
        Write-Host "==> Running Ruff linter..." -ForegroundColor Yellow
        & $VenvRuff check .
    }

    "typecheck" {
        Write-Host "==> Running Pyright static type checker..." -ForegroundColor Yellow
        & $VenvPyright
    }

    "audit" {
        Write-Host "==> Scanning dependencies with pip-audit..." -ForegroundColor Yellow
        & $VenvPython -m pip_audit -r requirements.txt
    }

    "run" {
        Write-Host "==> Starting RFM Cluster360 Streamlit Dashboard..." -ForegroundColor Yellow
        & $VenvStreamlit run app.py
    }

    "docker-build" {
        Write-Host "==> Building Docker image rfm-cluster360:test..." -ForegroundColor Yellow
        docker build -t rfm-cluster360:test .
    }

    "docker-run" {
        Write-Host "==> Launching container via Docker Compose..." -ForegroundColor Yellow
        docker compose up --build
    }

    "clean" {
        Write-Host "==> Cleaning cache directories..." -ForegroundColor Yellow
        Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path . -Recurse -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path . -Recurse -Directory -Filter ".ruff_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Cache cleaned." -ForegroundColor Green
    }
}
