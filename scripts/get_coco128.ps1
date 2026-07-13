$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$zipPath = Join-Path $root "coco128.zip"
$outDir = Join-Path $root "coco128"
$url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"

if (Test-Path $outDir) {
    Write-Host "coco128 already exists: $outDir"
    exit 0
}

Write-Host "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile $zipPath
Write-Host "Extracting $zipPath"
Expand-Archive -LiteralPath $zipPath -DestinationPath $root -Force
Remove-Item -LiteralPath $zipPath -Force
Write-Host "Done: $outDir"
