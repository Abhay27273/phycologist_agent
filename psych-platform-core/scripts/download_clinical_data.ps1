# Clinical Data Download Script for Windows
# Downloads WHO guidelines, clinical instruments, and other trusted resources

Write-Host "=== Downloading Clinical Resources ===" -ForegroundColor Cyan

# Create directory structure
Write-Host "`n[1/3] Creating directory structure..." -ForegroundColor Yellow
$directories = @(
    "data/guidelines/who",
    "data/guidelines/nice",
    "data/taxonomy",
    "data/instruments/phq9",
    "data/instruments/gad7",
    "data/instruments/cssrs"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Host "  ✓ Created: $dir" -ForegroundColor Green
}

# Download WHO Guidelines
Write-Host "`n[2/3] Downloading WHO Guidelines..." -ForegroundColor Yellow

try {
    Write-Host "  → WHO Mental Health Guidelines..." -NoNewline
    Invoke-WebRequest -Uri "https://www.who.int/publications/i/item/9789241549790" `
        -OutFile "data/guidelines/who/mental-health-gap-action-programme.html" `
        -UseBasicParsing
    Write-Host " ✓" -ForegroundColor Green
} catch {
    Write-Host " ✗ Failed: $_" -ForegroundColor Red
}

try {
    Write-Host "  → WHO ICD-11 Taxonomy..." -NoNewline
    Invoke-WebRequest -Uri "https://www.who.int/publications/i/item/9789240077263" `
        -OutFile "data/taxonomy/icd-11-reference.html" `
        -UseBasicParsing
    Write-Host " ✓" -ForegroundColor Green
} catch {
    Write-Host " ✗ Failed: $_" -ForegroundColor Red
}

# Download Clinical Instruments
Write-Host "`n[3/3] Downloading Clinical Instruments..." -ForegroundColor Yellow

try {
    Write-Host "  → PHQ-9 (Depression Screening)..." -NoNewline
    Invoke-WebRequest -Uri "https://integrationacademy.ahrq.gov/sites/default/files/2020-07/PHQ-9.pdf" `
        -OutFile "data/instruments/phq9/PHQ-9.pdf" `
        -UseBasicParsing
    Write-Host " ✓" -ForegroundColor Green
} catch {
    Write-Host " ✗ Failed: $_" -ForegroundColor Red
}

try {
    Write-Host "  → C-SSRS (Suicide Risk Screening)..." -NoNewline
    Invoke-WebRequest -Uri "https://www.cms.gov/files/document/cssrs-screen-version-instrument.pdf" `
        -OutFile "data/instruments/cssrs/cssrs-screen-version.pdf" `
        -UseBasicParsing
    Write-Host " ✓" -ForegroundColor Green
} catch {
    Write-Host " ✗ Failed: $_" -ForegroundColor Red
}

# Alternative sources for GAD-7
Write-Host "  → GAD-7 (Anxiety Screening)..." -NoNewline
try {
    # GAD-7 from PHQ Screeners site
    Invoke-WebRequest -Uri "https://www.phqscreeners.com/images/sites/g/files/g10016261/f/201412/GAD-7_English.pdf" `
        -OutFile "data/instruments/gad7/GAD-7.pdf" `
        -UseBasicParsing
    Write-Host " ✓" -ForegroundColor Green
} catch {
    Write-Host " ✗ Failed: $_" -ForegroundColor Red
}

Write-Host "`n=== Download Complete ===" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Review downloaded files in the data/ directory"
Write-Host "  2. Run: python scripts/ingest.py"
Write-Host "  3. This will embed all clinical resources into your RAG system"

# Summary
Write-Host "`nDownloaded Resources:" -ForegroundColor Cyan
Get-ChildItem -Path "data" -Recurse -File | ForEach-Object {
    $size = [math]::Round($_.Length / 1KB, 2)
    Write-Host "  • $($_.FullName.Replace((Get-Location).Path + '\', '')) ($size KB)" -ForegroundColor Gray
}
