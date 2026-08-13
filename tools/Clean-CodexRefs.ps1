$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Clean Codex Git Refs ===" -ForegroundColor Cyan
Write-Host ""

# 確認目前位於 Git Repository
$gitRoot = git rev-parse --show-toplevel 2>$null

if ($LASTEXITCODE -ne 0 -or -not $gitRoot) {
    Write-Host "ERROR: Current directory is not inside a Git repository." -ForegroundColor Red
    exit 1
}

$gitRoot = $gitRoot.Trim()

Write-Host "Repository:"
Write-Host $gitRoot
Write-Host ""

# 只搜尋 refs/codex
$refs = @(
    git for-each-ref --format="%(refname)" refs/codex
)

$refs = @(
    $refs |
    Where-Object { $_ -and $_.Trim() -ne "" } |
    ForEach-Object { $_.Trim() }
)

if ($refs.Count -eq 0) {
    Write-Host "No Codex refs found." -ForegroundColor Green
}
else {
    Write-Host "Found Codex refs:" -ForegroundColor Yellow

    foreach ($ref in $refs) {
        Write-Host "  $ref"
    }

    Write-Host ""
    Write-Host "Removing refs/codex only..." -ForegroundColor Yellow

    foreach ($ref in $refs) {

        # 額外安全檢查：
        # 只允許刪除 refs/codex/ 開頭的 ref
        if (-not $ref.StartsWith("refs/codex/")) {
            Write-Host "ERROR: Ref outside refs/codex detected: $ref" -ForegroundColor Red
            exit 1
        }

        git update-ref -d "$ref"

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to remove: $ref" -ForegroundColor Red
            exit 1
        }

        Write-Host "Removed: $ref" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Remaining refs/codex:"

$remaining = @(
    git for-each-ref --format="%(refname)" refs/codex
)

$remaining = @(
    $remaining |
    Where-Object { $_ -and $_.Trim() -ne "" }
)

if ($remaining.Count -eq 0) {
    Write-Host "  None" -ForegroundColor Green
}
else {
    foreach ($ref in $remaining) {
        Write-Host "  $ref" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Git status:"
git status --short

Write-Host ""
Write-Host "Done."