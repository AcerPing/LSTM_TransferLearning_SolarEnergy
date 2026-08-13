@echo off
chcp 65001 >nul
title Clean Codex Git Refs

REM Move to the project root (parent of this tools folder)
cd /d "%~dp0.."

echo.
echo ==========================================
echo   Clean Codex Git Refs for SourceTree
echo ==========================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Clean-CodexRefs.ps1"

echo.
echo ------------------------------------------
echo Finished. Press any key to close this window.
echo ------------------------------------------
pause >nul
