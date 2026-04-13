@echo off
chcp 65001 >nul
title Nocturne Data Forge
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [Nocturne] Создаю виртуальное окружение...
    python -m venv .venv
    if errorlevel 1 (
        echo Ошибка: не найден Python. Установите Python 3.10+ и добавьте в PATH.
        pause
        exit /b 1
    )
)

set "VENV_PY=.venv\Scripts\python.exe"
set "VENV_PIP=.venv\Scripts\pip.exe"
set "REQ_HASH_FILE=.venv\requirements.sha256"
for /f "delims=" %%h in ('powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 \"requirements.txt\").Hash"') do set "REQ_HASH=%%h"
set "OLD_REQ_HASH="
if exist "%REQ_HASH_FILE%" (
    set /p OLD_REQ_HASH=<"%REQ_HASH_FILE%"
)

if /I not "%REQ_HASH%"=="%OLD_REQ_HASH%" (
    echo [Nocturne] Обновляю зависимости...
    "%VENV_PIP%" install -r requirements.txt -q
    if errorlevel 1 (
        echo Ошибка установки зависимостей.
        pause
        exit /b 1
    )
    >"%REQ_HASH_FILE%" echo %REQ_HASH%
    echo [Nocturne] Зависимости обновлены.
) else (
    echo [Nocturne] Зависимости актуальны, пропускаю установку.
)

echo [Nocturne] Запуск...
"%VENV_PY%" main.py
if errorlevel 1 pause
exit /b 0
