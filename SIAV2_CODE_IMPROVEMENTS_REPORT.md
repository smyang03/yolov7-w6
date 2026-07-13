# SIAV2 Code Improvements Report

Date: 2026-07-13

## Scope

This pass applies code-safety improvements only. No training was run.

Implemented:

- Fixed `ComputeLossOTA` elementwise BCE initialization.
- Hardened aux-to-deploy checkpoint conversion.
- Added P3-lite/P4/P5 training cfg generation for `train_aux.py`.
- Verified P3-lite training/deploy cfg build and aux-to-deploy forward parity.

## 1. `ComputeLossOTA` Fix

File: `utils/loss.py`

Problem:

- `ComputeLossOTA.__init__` assigned `self.BCEcls_each` and `self.BCEobj_each` without defining `BCEcls_each` / `BCEobj_each`.
- This can raise `NameError` in the non-aux OTA path used by `train.py`.

Fix:

- Added `BCEWithLogitsLoss(..., reduction='none')` for class/objectness.
- Applied `FocalLoss` wrapping to those elementwise losses when `fl_gamma > 0`, matching `ComputeLossAuxOTA`.

Impact:

- `train_aux.py` SIAV2 path was already using `ComputeLossAuxOTA`.
- The fix removes a latent failure in the standard `train.py` OTA path.

## 2. Aux-To-Deploy Conversion Hardening

File: `tools/convert_aux_to_deploy.py`

Added strict checks:

- Non-head deploy weights must exist in the source checkpoint with matching shape.
- Deploy head count must match `deploy_det.nl`.
- Each fused source lead head must match the deploy Detect head weight/bias shape.
- Strict mode is default; `--no-strict` is available for debugging only.

Added optional forward verification:

```text
--verify-forward --verify-img 640
```

This compares the source aux model lead output with the converted deploy model output on a zero input. It catches cfg index drift and wrong head routing immediately.

## 3. P3-Lite Training CFG

File: `tools/make_siav2_p3_tradeoff_variants.py`

Added training cfg generation:

- `cfg/training/yolov7-l6-siav2-p3lite-p4p5-w250.yaml`

Training structure:

```text
lead: P3-lite / P4 / P5
aux:  P3 / P4 / P5
head: IAuxDetect
```

Lead detect layers:

- P3-lite: layer `77`
- P4: layer `78`
- P5: layer `89`

Aux detect layers:

- P3 aux: layer `90`
- P4 aux: layer `91`
- P5 aux: layer `92`

Final training head:

```text
[[77, 78, 89, 90, 91, 92], 1, IAuxDetect, [nc, anchors]]
```

## Verification

Completed:

```text
python -m py_compile utils/loss.py tools/convert_aux_to_deploy.py tools/make_siav2_p3_tradeoff_variants.py
```

Completed:

```text
conda run --no-capture-output -n yolov7 python models/yolo.py --cfg cfg/training/yolov7-l6-siav2-p3lite-p4p5-w250.yaml
conda run --no-capture-output -n yolov7 python models/yolo.py --cfg cfg/deploy/yolov7-l6-siav2-p3lite-p4p5-w250.yaml
```

Observed:

- Training cfg builds with `IAuxDetect`.
- Deploy cfg builds with `Detect`.

Completed:

```text
conda run --no-capture-output -n yolov7 python tools/make_random_weights.py --cfg cfg/training/yolov7-l6-siav2-p3lite-p4p5-w250.yaml --out weights/random-yolov7-l6-siav2-p3lite-p4p5-w250.pt --device 0 --seed 0
conda run --no-capture-output -n yolov7 python tools/convert_aux_to_deploy.py --weights weights/random-yolov7-l6-siav2-p3lite-p4p5-w250.pt --deploy-cfg cfg/deploy/yolov7-l6-siav2-p3lite-p4p5-w250.yaml --out weights/random-yolov7-l6-siav2-p3lite-p4p5-w250-deploy.pt --device 0 --verify-forward --verify-img 640
```

Observed:

- `copied_matching=480`
- `copied_detect_heads=3`
- `forward_parity_max_abs=0.00119019`

Completed:

```text
conda run --no-capture-output -n yolov7 python -c "<instantiate ComputeLossOTA and ComputeLossAuxOTA on P3-lite cfgs>"
```

Observed:

- `ComputeLossOTA`: `nl=3`, stride `[8, 16, 32]`
- `ComputeLossAuxOTA`: `nl=3`, stride `[8, 16, 32]`

No training script was run.
