@echo off
REM SPDX-License-Identifier: AGPL-3.0-or-later
REM Double-click helper: build the Windows .exe via the PowerShell script.
setlocal
pushd "%~dp0\.."
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\build_exe.ps1"
set RC=%ERRORLEVEL%
popd
pause
exit /b %RC%
