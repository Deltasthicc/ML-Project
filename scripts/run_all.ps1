# Full pipeline on Windows: every experiment -> tables -> report assets -> PDF. Resumable (finished runs are skipped).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 [-Workers 12]
param([int]$Workers = 12)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force results | Out-Null

python scripts/run_experiments.py --groups primary --native-eval --set "train.num_workers=$Workers" 2>&1 | Tee-Object -Append results/log_primary.txt
python scripts/run_experiments.py --groups factorial topology width --set "train.num_workers=$Workers" 2>&1 | Tee-Object -Append results/log_extra.txt

python scripts/aggregate_results.py
python scripts/auto_findings.py
python scripts/make_data_overview.py
python scripts/make_data_overview.py --strip --out results/figures/data_overview_strip.png
python scripts/make_report_assets.py
Push-Location report
try { pdflatex -interaction=nonstopmode main.tex; pdflatex -interaction=nonstopmode main.tex } catch { Write-Host "LaTeX failed/skipped - compile report/ on Overleaf" }
Pop-Location
Write-Host "done. Read results/findings.md, then write the \todo{} blocks in report/main.tex"
