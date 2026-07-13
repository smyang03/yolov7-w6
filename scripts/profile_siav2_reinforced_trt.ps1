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
New-Item -ItemType Directory -Force -Path $onnxDir, $trtDir, "weights" | Out-Null
Set-Content -Path "$trtDir\reinforced_coco80_trt_version.txt" -Value "TensorRT trtexec $trtVersion"

$siav2Train = "runs\siav2_coco128_train\siav2_p4small_aux_reinforced_1280_e100\weights\best.pt"
$siav2Deploy = "weights\siav2-p4small-aux-reinforced-coco128-deploy.pt"
$w6Train = "runs\siav2_coco128_train\w6_aux_1280_e100_rerun_throttle4s\weights\best.pt"
$w6Deploy = "weights\w6-coco128-rerun-throttle4s-deploy.pt"

conda run --no-capture-output -n yolov7 python tools\convert_aux_to_deploy.py `
    --weights $siav2Train `
    --deploy-cfg "cfg\deploy\yolov7-l6-siav2-p4small-coco80.yaml" `
    --out $siav2Deploy

conda run --no-capture-output -n yolov7 python tools\convert_aux_to_deploy.py `
    --weights $w6Train `
    --deploy-cfg "cfg\deploy\yolov7-w6.yaml" `
    --out $w6Deploy

conda run --no-capture-output -n yolov7 python tools\export_trt_onnx.py `
    --cfg "cfg\deploy\yolov7-l6-siav2-p4small-coco80.yaml" `
    --weights $siav2Deploy `
    --out "$onnxDir\siav2-p4small-reinforced-coco80-1280.onnx" `
    --img 1280 `
    --device 0 `
    --half `
    --no-constant-folding

conda run --no-capture-output -n yolov7 python tools\export_trt_onnx.py `
    --cfg "cfg\deploy\yolov7-w6.yaml" `
    --weights $w6Deploy `
    --out "$onnxDir\w6-coco80-rerun-1280.onnx" `
    --img 1280 `
    --device 0 `
    --half `
    --no-constant-folding

& $trtexec `
    --onnx="$onnxDir\siav2-p4small-reinforced-coco80-1280.onnx" `
    --fp16 `
    --saveEngine="$trtDir\siav2-p4small-reinforced-coco80-1280.fp16.engine" `
    --profilingVerbosity=detailed `
    --dumpProfile `
    --exportProfile="$trtDir\siav2-p4small-reinforced-coco80-1280.profile.json" `
    --exportLayerInfo="$trtDir\siav2-p4small-reinforced-coco80-1280.layers.json" `
    --warmUp=200 `
    --duration=1 `
    --iterations=50 `
    --avgRuns=10 `
    --noDataTransfers `
    --idleTime=20 `
    *> "$trtDir\siav2-p4small-reinforced-coco80-1280.trtexec.log"

& $trtexec `
    --onnx="$onnxDir\w6-coco80-rerun-1280.onnx" `
    --fp16 `
    --saveEngine="$trtDir\w6-coco80-rerun-1280.fp16.engine" `
    --profilingVerbosity=detailed `
    --dumpProfile `
    --exportProfile="$trtDir\w6-coco80-rerun-1280.profile.json" `
    --exportLayerInfo="$trtDir\w6-coco80-rerun-1280.layers.json" `
    --warmUp=200 `
    --duration=1 `
    --iterations=50 `
    --avgRuns=10 `
    --noDataTransfers `
    --idleTime=20 `
    *> "$trtDir\w6-coco80-rerun-1280.trtexec.log"

conda run --no-capture-output -n yolov7 python tools\summarize_trt_log.py `
    --log "$trtDir\siav2-p4small-reinforced-coco80-1280.trtexec.log" "$trtDir\w6-coco80-rerun-1280.trtexec.log" `
    --json-out "$trtDir\reinforced_coco80_trt_summary.json" `
    --csv-out "$trtDir\reinforced_coco80_trt_summary.csv"
