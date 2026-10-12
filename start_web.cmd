@echo off
cd /d "%~dp0"
if not exist "%~dp0venv\Scripts\python.exe" (
  echo Python environment missing: venv\Scripts\python.exe
  echo Create the virtual environment and install requirements first.
  pause
  exit /b 1
)
"%~dp0venv\Scripts\python.exe" "%~dp0launch_web.py"
if errorlevel 1 pause
