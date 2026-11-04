param(
    [Parameter(Mandatory = $true)]
    [string]$TargetRepo
)

$ErrorActionPreference = "Stop"
$source = $PSScriptRoot

if (-not (Test-Path (Join-Path $TargetRepo "algorithms"))) {
    throw "目标不是 QT-main 仓库根目录：$TargetRepo"
}

$files = @(
    "algorithms/technical_common.py",
    "algorithms/macd_trend.py",
    "algorithms/momentum_12_1.py",
    "algorithms/donchian_breakout.py"
)
foreach ($relative in $files) {
    $destination = Join-Path $TargetRepo $relative
    New-Item -ItemType Directory -Force (Split-Path $destination) | Out-Null
    Copy-Item (Join-Path $source $relative) $destination -Force
    Write-Host "updated $relative"
}

$datasets = @(
    "tech-us-10y",
    "manufacturing-us-10y",
    "healthcare-us-10y",
    "us-agri",
    "nasdaq100",
    "nasdaq-xbrl"
)
foreach ($dataset in $datasets) {
    $destination = Join-Path $TargetRepo "outdata/$dataset"
    New-Item -ItemType Directory -Force $destination | Out-Null
    Copy-Item (Join-Path $source "outdata/$dataset/*.json") $destination -Force
    Write-Host "updated outdata/$dataset"
}

Write-Host "完成。到目标仓库运行 git status 检查改动。"
