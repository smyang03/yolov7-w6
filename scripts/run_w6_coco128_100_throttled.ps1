$ErrorActionPreference = "Continue"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$env:WANDB_MODE = "disabled"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:256"
$env:YOLOV7_BATCH_SLEEP_MS = "4000"

New-Item -ItemType Directory -Force -Path "runs\siav2_coco128_train\logs" | Out-Null

conda run --no-capture-output -n yolov7 python train_aux.py `
    --workers 0 `
    --device 0 `
    --batch-size 2 `
    --img-size 1280 1280 `
    --epochs 100 `
    --data "data/coco128.yaml" `
    --project "runs/siav2_coco128_train" `
    --exist-ok `
    --close-mosaic 0 `
    --grad-clip 10 `
    --no-tensorboard `
    --noautoanchor `
    --notest `
    --cfg "cfg/training/yolov7-w6.yaml" `
    --weights "weights/random-yolov7-w6-coco80.pt" `
    --hyp "data/hyp.scratch.p6.yaml" `
    --name "w6_aux_1280_e100_rerun_throttle4s" `
    *> "runs\siav2_coco128_train\logs\w6_aux_1280_e100_rerun_throttle4s.log"
