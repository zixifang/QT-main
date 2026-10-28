@echo off
rem Double-click launcher for the GUI (Windows). ASCII only, so no mojibake.
rem
rem Usage:
rem   start-gui.cmd                      use .venv\Scripts\python.exe, else python on PATH
rem   start-gui.cmd <python.exe path>    use the given interpreter
rem
rem The GUI reads outdata/*.json only; it never opens data/*.csv.
setlocal
cd /d "%~dp0"

set "PY="
if not "%~1"=="" set "PY=%~1"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  echo [ERROR] No Python found.
  echo.
  echo Create the environment first:
  echo     python -m venv .venv
  echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

rem pip install may need UTF-8 mode on Chinese Windows; the GUI itself does not.
"%PY%" -c "import pandas, numpy, matplotlib" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] "%PY%" is missing pandas / numpy / matplotlib.
  echo.
  echo Install the dependencies:
  echo     "%PY%" -m pip install -r requirements.txt
  echo.
  echo If PowerShell blocks Activate.ps1, do not activate; call the interpreter directly.
  pause
  exit /b 1
)

rem outdata is committed, so this only runs on a fresh or cleaned tree.
if not exist "outdata\equity-2018\index.json" (
  echo [INFO] outdata is empty, generating results first...
  "%PY%" -m algorithms.rsi
  if errorlevel 1 goto :failed
)

echo Starting GUI with "%PY%" ...
"%PY%" -m gui
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo [ERROR] The command failed. See the message above.
pause
exit /b 1
