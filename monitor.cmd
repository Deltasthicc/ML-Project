@echo off
rem Double-click (or type "monitor" in a terminal at the repo root) to open the live study monitor in its own window.
cd /d "%~dp0"
".venv\python.exe" scripts\monitor.py --window %*
