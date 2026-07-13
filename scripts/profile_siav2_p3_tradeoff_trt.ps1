param(
    [string]$Trtexec = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\trtexec.exe"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
. "$root\scripts\siav2_trt_guard.ps1"

$trtexec = $Trtexec
$trtVersion = Assert-SIAV2TensorRTVersion -Trtexec $trtexec
$onnxDir = "runs\siav2\onnx"
$trtDir = "runs\siav2\trt"
New-Item -ItemType Directory -Force -Path $onnxDir, $trtDir | Out-Null
Set-Content -Path "$trtDir\p3_tradeoff_trt_version.txt" -Value "TensorRT trtexec $trtVersion"

conda run --no-capture-output -n yolov7 python tools\make_siav2_p3_tradeoff_variants.py

$models = @(
    @{
        Name = "siav2-p4p6-pruned-w250-nc16-1280"
        Cfg = "cfg\deploy\yolov7-l6-siav2-p4p6-pruned-w250.yaml"
    },
    @{
        Name = "siav2-p4p5-pruned-w250-nc16-1280"
        Cfg = "cfg\deploy\yolov7-l6-siav2-p4p5-pruned-w250.yaml"
    },
    @{
        Name = "siav2-p3lite-p4p6-w250-nc16-1280"
        Cfg = "cfg\deploy\yolov7-l6-siav2-p3lite-p4p6-w250.yaml"
    },
    @{
        Name = "siav2-p3lite-p4p5-w250-nc16-1280"
        Cfg = "cfg\deploy\yolov7-l6-siav2-p3lite-p4p5-w250.yaml"
    },
    @{
        Name = "siav2-p3full-p4p5-w250-nc16-1280"
        Cfg = "cfg\deploy\yolov7-l6-siav2-p3full-p4p5-w250.yaml"
    }
)

$logs = @()
foreach ($model in $models) {
    $name = $model.Name
    $cfg = $model.Cfg
    $onnx = "$onnxDir\$name.onnx"
    $engine = "$trtDir\$name.fp16.engine"
    $log = "$trtDir\$name.trtexec.log"
    $logs += $log

    conda run --no-capture-output -n yolov7 python tools\export_trt_onnx.py `
        --cfg $cfg `
        --out $onnx `
        --img 1280 `
        --device 0 `
        --half `
        --no-constant-folding

    & $trtexec `
        --onnx="$onnx" `
        --fp16 `
        --saveEngine="$engine" `
        --profilingVerbosity=detailed `
        --dumpProfile `
        --exportProfile="$trtDir\$name.profile.json" `
        --exportLayerInfo="$trtDir\$name.layers.json" `
        --warmUp=200 `
        --duration=1 `
        --iterations=50 `
        --avgRuns=10 `
        --noDataTransfers `
        --idleTime=20 `
        *> $log
}

conda run --no-capture-output -n yolov7 python tools\summarize_trt_log.py `
    --log $logs `
    --json-out "$trtDir\p3_tradeoff_trt_summary.json" `
    --csv-out "$trtDir\p3_tradeoff_trt_summary.csv"

conda run --no-capture-output -n yolov7 python tools\analyze_trt_reformat.py `
    --log $logs `
    --json-out "$trtDir\p3_tradeoff_reformat_summary.json" `
    --csv-out "$trtDir\p3_tradeoff_reformat_summary.csv"
