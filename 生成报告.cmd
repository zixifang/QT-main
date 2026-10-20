@echo off
chcp 65001 >nul
cd /d "%~dp0"
".venv\Scripts\python.exe" -m services.demo
if errorlevel 1 goto fail
for /f "delims=" %%i in ('dir /b /o-d "reports\ndx_enhance_*.html"') do (
  start "" "%~dp0reports\%%i"
  goto done
)
:done
echo.
echo Report opened. Press any key to close.
pause >nul
exit /b 0
:fail
echo.
echo Report failed. Press any key to close.
pause >nul
exit /b 1
