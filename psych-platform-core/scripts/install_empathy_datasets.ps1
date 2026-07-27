# ============================================================
# Installer: Psychologist Emotional-Understanding & Counseling Datasets
# Windows PowerShell Version
# Date: 2026-02-05
# ============================================================

$ErrorActionPreference = "Stop"
$ROOT = "datasets"

# Create root directory
New-Item -ItemType Directory -Force -Path $ROOT | Out-Null
Set-Location $ROOT

Write-Host "=== Creating top-level directories ===" -ForegroundColor Cyan
$directories = @("empathetic_dialogues", "pysdial", "mentalchat16k", "medic", "mh_counseling", "counselchat", "ecc", "logs")
foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

function Log {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage -ForegroundColor Green
    
    # Ensure logs directory exists
    $logDir = "../logs"
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }
    Add-Content -Path "$logDir/install.log" -Value $logMessage
}

# ------------------------------------------------------------
# 1) EmpatheticDialogues (ED)
# ------------------------------------------------------------
Log "EmpatheticDialogues (ED) ..."
Set-Location empathetic_dialogues
if (-not (Test-Path "empatheticdialogues.tar.gz")) {
    Write-Host "  Downloading EmpatheticDialogues..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz" `
        -OutFile "empatheticdialogues.tar.gz" -UseBasicParsing
}
Write-Host "  EmpatheticDialogues downloaded (extract manually with 7-Zip or tar)" -ForegroundColor Green
Set-Location ..

# ------------------------------------------------------------
# 2) PsyDial (privacy-preserving long-term counseling dialogs)
# ------------------------------------------------------------
Log "PsyDial (repo clone) ..."
Set-Location pysdial
if (-not (Test-Path "PsyDial")) {
    Write-Host "  Cloning PsyDial repository..." -ForegroundColor Yellow
    git clone https://github.com/qiuhuachuan/PsyDial.git
}
Set-Location ..

# ------------------------------------------------------------
# 3) MentalChat-16K (HF dataset)
# ------------------------------------------------------------
Log "MentalChat-16K (Hugging Face) ..."
Set-Location mentalchat16k
if (Get-Command huggingface-cli -ErrorAction SilentlyContinue) {
    Write-Host "  Downloading MentalChat-16K..." -ForegroundColor Yellow
    huggingface-cli download ShenLab/MentalChat16K --local-dir . --repo-type dataset
} else {
    $content = @"
MentalChat-16K requires Hugging Face CLI for convenient download.

Quick steps:
  pip install -U "huggingface_hub[cli]"
  huggingface-cli download ShenLab/MentalChat16K --local-dir . --repo-type dataset

Homepage:
  https://huggingface.co/datasets/ShenLab/MentalChat16K
Paper:
  https://arxiv.org/abs/2503.13509
"@
    $content | Out-File -FilePath "HOW_TO_GET.txt" -Encoding UTF8
    Write-Host "  Created HOW_TO_GET.txt with instructions" -ForegroundColor Yellow
}
Set-Location ..

# ------------------------------------------------------------
# 4) MEDIC (Multimodal Empathy in Psychotherapy)
# ------------------------------------------------------------
Log "MEDIC requires access request (creating placeholder) ..."
Set-Location medic
$content = @"
MEDIC (Multimodal Empathy in Counseling) dataset typically requires permission from authors.

Official page:
  https://ustc-ac.github.io/datasets/medic/

Request access as instructed on the page. Once granted, place files here.
"@
$content | Out-File -FilePath "READ_ME_FIRST.txt" -Encoding UTF8
Set-Location ..

# ------------------------------------------------------------
# 5) Mental Health Counseling Conversations (HF)
# ------------------------------------------------------------
Log "Mental Health Counseling Conversations (Hugging Face) ..."
Set-Location mh_counseling
if (Get-Command huggingface-cli -ErrorAction SilentlyContinue) {
    Write-Host "  Downloading Mental Health Counseling..." -ForegroundColor Yellow
    huggingface-cli download Amod/mental_health_counseling_conversations --local-dir . --repo-type dataset
} else {
    $content = @"
This dataset is hosted on Hugging Face.

Install CLI:
  pip install -U "huggingface_hub[cli]"

Download:
  huggingface-cli download Amod/mental_health_counseling_conversations --local-dir . --repo-type dataset

Card:
  https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
"@
    $content | Out-File -FilePath "HOW_TO_GET.txt" -Encoding UTF8
    Write-Host "  Created HOW_TO_GET.txt with instructions" -ForegroundColor Yellow
}
Set-Location ..

# ------------------------------------------------------------
# 6) CounselChat (therapist Q&A)
# ------------------------------------------------------------
Log "CounselChat (GitHub repo) ..."
Set-Location counselchat
if (-not (Test-Path "counsel-chat")) {
    Write-Host "  Cloning CounselChat repository..." -ForegroundColor Yellow
    git clone https://github.com/nbertagnolli/counsel-chat.git
}
$content = @"
CounselChat dataset is mirrored on Hugging Face as:
  nbertagnolli/counsel-chat

To pull via HF:
  huggingface-cli download nbertagnolli/counsel-chat --local-dir ./hf_mirror --repo-type dataset
"@
$content | Out-File -FilePath "OPTIONAL.txt" -Encoding UTF8
Set-Location ..

# ------------------------------------------------------------
# 7) ECC (Emotion-Cause Conversation)
# ------------------------------------------------------------
Log "ECC (Emotion-Cause Conversations) ..."
Set-Location ecc
if (-not (Test-Path "ECC")) {
    Write-Host "  Cloning ECC repository..." -ForegroundColor Yellow
    try {
        git clone https://github.com/Yuan-23/ECC.git
    } catch {
        Write-Host "  ECC repo clone failed (may not exist)" -ForegroundColor Yellow
    }
}
if (-not (Test-Path "ECC_paper_emnlp2025.pdf")) {
    Write-Host "  Downloading ECC paper..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest -Uri "https://aclanthology.org/2025.emnlp-main.306.pdf" `
            -OutFile "ECC_paper_emnlp2025.pdf" -UseBasicParsing
    } catch {
        Write-Host "  Paper download failed" -ForegroundColor Yellow
    }
}
Set-Location ..

Log "All steps completed. Review ./logs/install.log for details."

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "Datasets downloaded to: $((Get-Location).Path)\datasets" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Install Hugging Face CLI: pip install -U 'huggingface_hub[cli]'" -ForegroundColor White
Write-Host "  2. Download HF datasets using instructions in HOW_TO_GET.txt files" -ForegroundColor White
Write-Host "  3. Extract .tar.gz files using 7-Zip or tar command" -ForegroundColor White
Write-Host "  4. Review logs/install.log for details" -ForegroundColor White
