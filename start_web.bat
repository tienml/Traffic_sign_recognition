@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py serve %*
) else (
    python main.py serve %*
)
if errorlevel 1 (
    echo.
    echo Khong khoi dong duoc web. Xem huong dan cai dat trong README.md.
    pause
)
endlocal
