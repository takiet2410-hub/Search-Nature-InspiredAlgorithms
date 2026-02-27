param(
    [string]$InputFile = "algorithm_explanations.md",
    [string]$OutputFile = "algorithm_explanations.pdf"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $InputFile)) {
    Write-Error "Input file not found: $InputFile"
}

$pandoc = Get-Command pandoc -ErrorAction SilentlyContinue
if (-not $pandoc) {
    Write-Host "Pandoc is not installed."
    Write-Host "Install Pandoc: https://pandoc.org/installing.html"
    Write-Host "Then rerun:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/export_report.ps1"
    exit 1
}

try {
    pandoc $InputFile -o $OutputFile --from markdown --toc
    Write-Host "PDF exported successfully: $OutputFile"
}
catch {
    Write-Error "Failed to export PDF. Details: $($_.Exception.Message)"
}
