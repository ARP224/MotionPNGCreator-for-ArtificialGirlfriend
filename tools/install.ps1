# =============================================================================
# MotionPNGCreator for ArtificialGirlfriend setup script
#
# Normally run via "Installer MotionPNGCreator for AG.bat" in the repository
# root. To run directly:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\install.ps1
#
# Safe to run multiple times (already-installed items are skipped).
# Options: -Lang ja|en (default: OS display language), -NoPause (for testing)
# =============================================================================
param(
    [switch]$NoPause,
    [ValidateSet('ja', 'en')][string]$Lang
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($Lang) {
    $script:ja = ($Lang -eq 'ja')
} else {
    $script:ja = ([System.Globalization.CultureInfo]::CurrentUICulture.TwoLetterISOLanguageName -eq 'ja')
}

# Pick the Japanese or English message depending on the OS display language
function T([string]$jaText, [string]$enText) {
    if ($script:ja) { $jaText } else { $enText }
}

$installerName = 'Installer MotionPNGCreator for AG.bat'

function Wait-Enter {
    if (-not $NoPause) {
        Read-Host (T 'Enter キーを押すと終了します' 'Press Enter to exit') | Out-Null
    }
}

# Rebuild PATH from the registry so tools installed moments ago are usable
# in this same window
function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Test-Cmd([string]$name) {
    [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Abort([string]$message) {
    Write-Host ''
    Write-Host ((T '[エラー] ' '[ERROR] ') + $message) -ForegroundColor Red
    Wait-Enter
    exit 1
}

function Install-WithWinget([string]$id, [string]$label) {
    Write-Host ((T "[$label] インストールします（winget: $id）..." "[$label] Installing (winget: $id)..."))
    winget install -e --id $id --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Abort (T "$label のインストールに失敗しました（winget 終了コード: $LASTEXITCODE）。ネットワーク接続を確認して「$installerName」を再実行してください。" `
                 "Failed to install $label (winget exit code: $LASTEXITCODE). Check your network connection and run `"$installerName`" again.")
    }
    Update-SessionPath
}

function Confirm-Installed([string]$command, [string]$label) {
    if (-not (Test-Cmd $command)) {
        Abort (T "$label をインストールしましたが認識できません。PCを再起動してから「$installerName」を再実行してください。" `
                 "$label was installed but is not recognized yet. Restart your PC and run `"$installerName`" again.")
    }
}

Write-Host '============================================================'
Write-Host ' MotionPNGCreator for ArtificialGirlfriend Setup'
Write-Host '============================================================'
Write-Host ''
Write-Host (T '数GBのダウンロードを含むため、完了まで時間がかかることがあります。' `
             'This setup includes several GB of downloads and may take a while.')
Write-Host (T "途中で失敗しても「$installerName」を再実行すれば続きから進みます。" `
             "If it fails midway, run `"$installerName`" again to continue from where it left off.")
Write-Host ''

# --- 0. winget (standard Windows package manager; the only prerequisite) ---
if (-not (Test-Cmd winget)) {
    Abort (T "winget が見つかりません。Microsoft Store で「アプリ インストーラー」をインストール（または更新）してから、「$installerName」を再実行してください。" `
             "winget was not found. Install (or update) `"App Installer`" from the Microsoft Store, then run `"$installerName`" again.")
}

# --- 1. Git (uv sync needs it to fetch the CLIP dependency from GitHub) ---
if (Test-Cmd git) {
    Write-Host (T '[Git] 導入済み → スキップ' '[Git] Already installed -> skip')
} else {
    Install-WithWinget 'Git.Git' 'Git'
    Confirm-Installed 'git' 'Git'
}

# --- 2. ffmpeg (needed for WebM output, audio mux, and previews) ---
if (Test-Cmd ffmpeg) {
    Write-Host (T '[ffmpeg] 導入済み → スキップ' '[ffmpeg] Already installed -> skip')
} else {
    Install-WithWinget 'Gyan.FFmpeg' 'ffmpeg'
    Confirm-Installed 'ffmpeg' 'ffmpeg'
}

# --- 3. Node.js (needed to launch the Electron player) ---
if (Test-Cmd npm) {
    Write-Host (T '[Node.js] 導入済み → スキップ' '[Node.js] Already installed -> skip')
} else {
    Install-WithWinget 'OpenJS.NodeJS.LTS' 'Node.js'
    Confirm-Installed 'npm' 'Node.js'
}

# --- 4. uv (manages the Python runtime and packages) ---
if (Test-Cmd uv) {
    Write-Host (T '[uv] 導入済み → スキップ' '[uv] Already installed -> skip')
} else {
    Write-Host (T '[uv] インストールします...' '[uv] Installing...')
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    } catch {
        Abort (T "uv のインストールに失敗しました: $($_.Exception.Message)" `
                 "Failed to install uv: $($_.Exception.Message)")
    }
    Update-SessionPath
    if (-not (Test-Cmd uv)) {
        # Add the default install location directly and check again
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
        Confirm-Installed 'uv' 'uv'
    }
}

# --- 5. Python packages (PyTorch CUDA etc.; the longest step) ---
Write-Host ''
Write-Host (T '[Python環境] uv sync を実行します（初回は数GBのダウンロード。しばらくお待ちください）...' `
             '[Python] Running uv sync (first run downloads several GB; please wait)...')
& uv sync
if ($LASTEXITCODE -ne 0) {
    Abort (T "uv sync に失敗しました。上のエラー内容を確認し、「$installerName」を再実行してください（ダウンロード済みの部分はキャッシュされ、続きから進みます）。" `
             "uv sync failed. Check the error above and run `"$installerName`" again (downloaded parts are cached, so it continues from where it left off).")
}

# --- 6. Electron player dependencies (npm install) ---
Write-Host ''
Write-Host (T '[プレイヤー] npm install を実行します...' '[Player] Running npm install...')
Push-Location (Join-Path $repoRoot 'MotionPNGTuber_Player')
& npm install
$npmExit = $LASTEXITCODE
Pop-Location
if ($npmExit -ne 0) {
    Abort (T "npm install に失敗しました。ネットワーク接続を確認して「$installerName」を再実行してください。" `
             "npm install failed. Check your network connection and run `"$installerName`" again.")
}

# --- Completion summary ---
Write-Host ''
if (-not (Test-Cmd nvidia-smi)) {
    Write-Host (T '[注意] NVIDIA ドライバが見つかりません。本ツールは NVIDIA GPU（CUDA）が必須です。' `
                 '[NOTE] NVIDIA driver not found. This tool requires an NVIDIA GPU (CUDA).') -ForegroundColor Yellow
    Write-Host (T '       CPUでの動作は未確認・サポート対象外です。' `
                 '       Running on the CPU is untested and unsupported.') -ForegroundColor Yellow
    Write-Host ''
}
Write-Host '============================================================' -ForegroundColor Green
Write-Host (T ' セットアップが完了しました！' ' Setup completed!') -ForegroundColor Green
Write-Host '============================================================' -ForegroundColor Green
Write-Host ''
Write-Host (T '残りの手動作業:' 'Remaining manual steps:')
Write-Host (T '  1. SAM3モデルを Sam3\sam3.pt に配置（入手方法: Sam3\README.txt）' `
             '  1. Place the SAM3 model at Sam3\sam3.pt (see Sam3\README.txt)')
Write-Host (T '  2. 動画生成する場合は Vidu APIキーを用意（起動後、タブ⓪で設定）' `
             '  2. To generate videos, get a Vidu API key (set it in tab 0 after launch)')
Write-Host ''
Write-Host (T '起動方法（ダブルクリック）:' 'How to launch (double-click):')
Write-Host (T '  Start_Asset_Preparer(step1).bat  … 素材準備GUI（Step 1）' `
             '  Start_Asset_Preparer(step1).bat  ... asset preparation GUI (Step 1)')
Write-Host (T '  Start_Video_Generator(step2).bat … バッチ自動生成GUI（Step 2）' `
             '  Start_Video_Generator(step2).bat ... batch generation GUI (Step 2)')
Write-Host ''
Wait-Enter
exit 0
