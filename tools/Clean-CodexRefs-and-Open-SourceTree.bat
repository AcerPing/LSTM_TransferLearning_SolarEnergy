@echo off
chcp 65001 >nul
setlocal
title Clean Codex Refs and Open SourceTree

REM Move to project root (parent of tools folder)
cd /d "%~dp0.."

echo.
echo ==========================================
echo   Clean Codex Refs and Open SourceTree
echo ==========================================
echo.

REM Run cleanup script
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Clean-CodexRefs.ps1"
set "CLEAN_EXIT=%ERRORLEVEL%"

echo.

if not "%CLEAN_EXIT%"=="0" (
    echo [ERROR] Clean-CodexRefs.ps1 failed with exit code %CLEAN_EXIT%.
    echo SourceTree will NOT be opened.
    echo.
    pause
    exit /b %CLEAN_EXIT%
)

echo [OK] Codex refs cleanup completed successfully.
echo.

REM Try common SourceTree install locations
set "SOURCETREE_EXE="

if exist "%LocalAppData%\SourceTree\SourceTree.exe" (
    set "SOURCETREE_EXE=%LocalAppData%\SourceTree\SourceTree.exe"
)

if not defined SOURCETREE_EXE if exist "%LocalAppData%\Atlassian\SourceTree\SourceTree.exe" (
    set "SOURCETREE_EXE=%LocalAppData%\Atlassian\SourceTree\SourceTree.exe"
)

if not defined SOURCETREE_EXE if exist "%ProgramFiles%\Atlassian\SourceTree\SourceTree.exe" (
    set "SOURCETREE_EXE=%ProgramFiles%\Atlassian\SourceTree\SourceTree.exe"
)

if not defined SOURCETREE_EXE if defined ProgramFiles(x86) if exist "%ProgramFiles(x86)%\Atlassian\SourceTree\SourceTree.exe" (
    set "SOURCETREE_EXE=%ProgramFiles(x86)%\Atlassian\SourceTree\SourceTree.exe"
)

if not defined SOURCETREE_EXE (
    where SourceTree.exe >nul 2>&1
    if "%ERRORLEVEL%"=="0" (
        set "SOURCETREE_EXE=SourceTree.exe"
    )
)

if not defined SOURCETREE_EXE (
    echo [ERROR] SourceTree.exe could not be found automatically.
    echo Codex refs were cleaned successfully, but SourceTree was not opened.
    echo.
    echo You can edit this BAT later and set SOURCETREE_EXE to your SourceTree.exe path.
    echo.
    pause
    exit /b 2
)

echo Opening SourceTree:
echo %SOURCETREE_EXE%
echo.

start "" "%SOURCETREE_EXE%"

if errorlevel 1 (
    echo [ERROR] Failed to launch SourceTree.
    echo.
    pause
    exit /b 3
)

echo [OK] SourceTree launch requested.
timeout /t 2 /nobreak >nul
exit /b 0
