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
Set-Content -Path "$trtDir\nc16_rawhead_trt_version.txt" -Value "TensorRT trtexec $trtVersion"

$siav2Cfg = "cfg\deploy\yolov7-l6-siav2-p4p6-pruned-w250.yaml"
$w6Cfg = "cfg\deploy\yolov7-w6-nc16.yaml"

conda run --no-capture-output -n yolov7 python tools\export_trt_onnx.py `
    --cfg $siav2Cfg `
    --out "$onnxDir\siav2-p4p6-pruned-w250-nc16-decoded-1280.onnx" `
    --img 1280 `
    --device 0 `
    --half `
    --no-constant-folding

conda run --no-capture-output -n yolov7 python tools\export_trt_onnx.py `
    --cfg $siav2Cfg `
    --out "$onnxDir\siav2-p4p6-pruned-w250-nc16-raw-1280.onnx" `
    --img 1280 `
    --device 0 `
    --half `
    --raw-head `
    --no-constant-folding

conda run --no-capture-output -n yolov7 python tools\export_trt_onnx.py `
    --cfg $w6Cfg `
    --out "$onnxDir\w6-nc16-decoded-1280.onnx" `
    --img 1280 `
    --device 0 `
    --half `
    --no-constant-folding

conda run --no-capture-output -n yolov7 python tools\export_trt_onnx.py `
    --cfg $w6Cfg `
    --out "$onnxDir\w6-nc16-raw-1280.onnx" `
    --img 1280 `
    --device 0 `
    --half `
    --raw-head `
    --no-constant-folding

& $trtexec `
    --onnx="$onnxDir\siav2-p4p6-pruned-w250-nc16-decoded-1280.onnx" `
    --fp16 `
    --saveEngine="$trtDir\siav2-p4p6-pruned-w250-nc16-decoded-1280.fp16.engine" `
    --profilingVerbosity=detailed `
    --dumpProfile `
    --exportProfile="$trtDir\siav2-p4p6-pruned-w250-nc16-decoded-1280.profile.json" `
    --exportLayerInfo="$trtDir\siav2-p4p6-pruned-w250-nc16-decoded-1280.layers.json" `
    --warmUp=200 `
    --duration=1 `
    --iterations=50 `
    --avgRuns=10 `
    --noDataTransfers `
    --idleTime=20 `
    *> "$trtDir\siav2-p4p6-pruned-w250-nc16-decoded-1280.trtexec.log"

& $trtexec `
    --onnx="$onnxDir\siav2-p4p6-pruned-w250-nc16-raw-1280.onnx" `
    --fp16 `
    --saveEngine="$trtDir\siav2-p4p6-pruned-w250-nc16-raw-1280.fp16.engine" `
    --profilingVerbosity=detailed `
    --dumpProfile `
    --exportProfile="$trtDir\siav2-p4p6-pruned-w250-nc16-raw-1280.profile.json" `
    --exportLayerInfo="$trtDir\siav2-p4p6-pruned-w250-nc16-raw-1280.layers.json" `
    --warmUp=200 `
    --duration=1 `
    --iterations=50 `
    --avgRuns=10 `
    --noDataTransfers `
    --idleTime=20 `
    *> "$trtDir\siav2-p4p6-pruned-w250-nc16-raw-1280.trtexec.log"

& $trtexec `
    --onnx="$onnxDir\w6-nc16-decoded-1280.onnx" `
    --fp16 `
    --saveEngine="$trtDir\w6-nc16-decoded-1280.fp16.engine" `
    --profilingVerbosity=detailed `
    --dumpProfile `
    --exportProfile="$trtDir\w6-nc16-decoded-1280.profile.json" `
    --exportLayerInfo="$trtDir\w6-nc16-decoded-1280.layers.json" `
    --warmUp=200 `
    --duration=1 `
    --iterations=50 `
    --avgRuns=10 `
    --noDataTransfers `
    --idleTime=20 `
    *> "$trtDir\w6-nc16-decoded-1280.trtexec.log"

& $trtexec `
    --onnx="$onnxDir\w6-nc16-raw-1280.onnx" `
    --fp16 `
    --saveEngine="$trtDir\w6-nc16-raw-1280.fp16.engine" `
    --profilingVerbosity=detailed `
    --dumpProfile `
    --exportProfile="$trtDir\w6-nc16-raw-1280.profile.json" `
    --exportLayerInfo="$trtDir\w6-nc16-raw-1280.layers.json" `
    --warmUp=200 `
    --duration=1 `
    --iterations=50 `
    --avgRuns=10 `
    --noDataTransfers `
    --idleTime=20 `
    *> "$trtDir\w6-nc16-raw-1280.trtexec.log"

conda run --no-capture-output -n yolov7 python tools\summarize_trt_log.py `
    --log "$trtDir\siav2-p4p6-pruned-w250-nc16-decoded-1280.trtexec.log" "$trtDir\siav2-p4p6-pruned-w250-nc16-raw-1280.trtexec.log" "$trtDir\w6-nc16-decoded-1280.trtexec.log" "$trtDir\w6-nc16-raw-1280.trtexec.log" `
    --json-out "$trtDir\nc16_rawhead_trt_summary.json" `
    --csv-out "$trtDir\nc16_rawhead_trt_summary.csv"

conda run --no-capture-output -n yolov7 python tools\analyze_trt_reformat.py `
    --log "$trtDir\siav2-p4p6-pruned-w250-nc16-decoded-1280.trtexec.log" "$trtDir\siav2-p4p6-pruned-w250-nc16-raw-1280.trtexec.log" "$trtDir\w6-nc16-decoded-1280.trtexec.log" "$trtDir\w6-nc16-raw-1280.trtexec.log" `
    --json-out "$trtDir\nc16_rawhead_reformat_summary.json" `
    --csv-out "$trtDir\nc16_rawhead_reformat_summary.csv"
