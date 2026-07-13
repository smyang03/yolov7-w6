$ErrorActionPreference = "Continue"

$env:WANDB_MODE = "disabled"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:256"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
New-Item -ItemType Directory -Force -Path "runs\siav2_coco128_train\logs" | Out-Null

$common = @(
    "train_aux.py",
    "--workers", "0",
    "--device", "0",
    "--batch-size", "2",
    "--img-size", "1280", "1280",
    "--epochs", "100",
    "--data", "data/coco128.yaml",
    "--project", "runs/siav2_coco128_train",
    "--exist-ok",
    "--grad-clip", "10",
    "--no-tensorboard",
    "--noautoanchor",
    "--notest",
    "--cfg", "cfg/training/yolov7-l6-siav2-p4small-coco80.yaml",
    "--weights", "weights/random-yolov7-l6-siav2-p4small-coco80.pt"
)

conda run --no-capture-output -n yolov7 python @common `
    --name "siav2_p4small_aux_1280_e100" `
    --hyp "data/hyp.siav2-p4small-aux.yaml" `
    --close-mosaic "15" *> "runs\siav2_coco128_train\logs\siav2_p4small_aux_1280_e100.log"

conda run --no-capture-output -n yolov7 python @common `
    --name "siav2_p4small_aux_conservative_1280_e100" `
    --hyp "data/hyp.siav2-p4small-aux-conservative.yaml" `
    --close-mosaic "15" *> "runs\siav2_coco128_train\logs\siav2_p4small_aux_conservative_1280_e100.log"

conda run --no-capture-output -n yolov7 python @common `
    --name "siav2_p4small_aux_relaxed_1280_e100" `
    --hyp "data/hyp.siav2-p4small-aux-relaxed.yaml" `
    --close-mosaic "20" *> "runs\siav2_coco128_train\logs\siav2_p4small_aux_relaxed_1280_e100.log"
