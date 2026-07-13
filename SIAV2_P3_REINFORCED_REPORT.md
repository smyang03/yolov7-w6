# SIAV2 P3 Removal Reinforcement Report

Date: 2026-07-13

## 1. P3 removal mitigation

P3 was not restored in the final inference graph. The mitigation is to make P4 absorb small-object responsibility while keeping the deploy graph at P4/P5/P6 only.

Implemented mitigation:

- P4 small anchors and small-object fallback assignment remain the main P3 replacement.
- Optional small-object loss weighting was added for box/cls/obj on small targets.
- Optional small-object dynamic-k floor was added so small targets receive more matching candidates.
- Aux heads are used during training only; deploy uses the 3-head `Detect` graph.
- Final TRT graph still has no P3 branch, so the mitigation does not add inference latency.

Important result: the stronger `reinforced` hyp is not selected as the default training recipe because it underperformed the previous `relaxed` hyp on COCO128 100epoch smoke training. The final architecture remains P4/P5/P6, but the default training hyp should stay relaxed unless the real SIAV2 dataset proves the extra small-object weighting helps.

## 2. Code and files added/changed

- `utils/loss.py`
  - Added optional small-object box/cls/obj weighting.
  - Added optional small-object dynamic-k minimum.
  - Kept settings controlled by hyp keys, so the behavior is off unless enabled.
- `data/hyp.siav2-p4small-aux-reinforced.yaml`
  - Enables the extra small-object reinforcement branch:
    - `small_obj_loss: 1`
    - `small_obj_loss_max_px: 128`
    - `small_obj_box_gain: 1.5`
    - `small_obj_cls_gain: 1.25`
    - `small_obj_obj_gain: 2.0`
    - `small_obj_aux_gain: 1.25`
    - `small_obj_min_dynamic_k: 3`
- `train_aux.py`
  - Added optional `YOLOV7_BATCH_SLEEP_MS` batch sleep throttle.
  - Default is `0`, so normal training is unchanged.
- `scripts/run_w6_coco128_100_throttled.ps1`
  - W6 rerun script with `YOLOV7_BATCH_SLEEP_MS=4000` to keep average GPU utilization under the requested limit.
- `tools/convert_aux_to_deploy.py`
  - Converts `IAuxDetect` training checkpoints into deploy `Detect` checkpoints.
- `tools/export_trt_onnx.py`
  - Added `--weights`.
  - Added `--no-constant-folding`.
- `scripts/profile_siav2_reinforced_trt.ps1`
  - Converts checkpoints, exports ONNX, builds TensorRT FP16 engines, and writes layer profiles.
- `tools/summarize_trt_log.py`
  - Summarizes `trtexec --dumpProfile` logs into JSON/CSV.
- `tools/check_siav2_loss.py`
  - Synthetic loss smoke test for the reinforced SIAV2 aux loss path.

## 3. Verification

Passed:

- `python -m py_compile utils/loss.py train_aux.py models/yolo.py`
- `python tools/check_siav2_loss.py`
- 1epoch real dataloader/optimizer smoke for SIAV2 reinforced
- SIAV2 reinforced COCO128 100epoch training
- W6 COCO128 100epoch rerun with GPU throttle
- Aux-to-deploy conversion for SIAV2 and W6
- ONNX export for SIAV2 and W6
- TensorRT FP16 build/profile for SIAV2 and W6

## 4. COCO128 100epoch results

These are random-init COCO128 smoke results, not production quality results.

| Model | Train cfg/hyp | Image | Final box | Final obj | Final cls | Final total | P | R | mAP50 | mAP50-95 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SIAV2 P4small aux relaxed | `hyp.siav2-p4small-aux-relaxed.yaml` | 1280 | 0.08081 | 0.13510 | 0.05934 | 0.27520 | 0.288 | 0.008362 | 0.002400 | 0.0004867 | Best smoke metric, keep as default |
| SIAV2 P4small aux reinforced | `hyp.siav2-p4small-aux-reinforced.yaml` | 1280 | 0.08732 | 0.17430 | 0.06111 | 0.32270 | 0.847 | 0.001083 | 0.0003228 | 0.0000768 | Optional, not default |
| YOLOv7-W6 aux rerun throttle4s | `hyp.scratch.p6.yaml` | 1280 | 0.08881 | 0.04895 | 0.06712 | 0.20490 | 3.152e-05 | 0.003050 | 8.152e-06 | 1.254e-06 | Baseline |

Interpretation:

- Reinforced is better than W6 on mAP50/mAP50-95 in this smoke run.
- Reinforced is worse than relaxed on mAP50/mAP50-95.
- Therefore the extra small-object loss/dynamic-k reinforcement is not selected as the default.
- P3 compensation should remain the relaxed P4-small assignment/anchor strategy unless real SIAV2 data shows a different result.

## 5. TRT FP16 latency

Final generated summary: `runs/siav2/trt/reinforced_coco80_trt_summary.csv`

| Model | Heads | Output | Avg ms | Median ms | Backbone ms | Neck ms | Detect/decode ms | Speedup vs W6 avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIAV2 P4small reinforced deploy | P4/P5/P6 | decoded `(1,25200,85)` | 1.6241 | 1.4601 | 0.6904 | 0.5969 | 0.3374 | 2.46x |
| YOLOv7-W6 deploy | P3/P4/P5/P6 | decoded `(1,102000,85)` | 3.9961 | 3.7349 | 1.7390 | 1.6712 | 0.5321 | 1.00x |

Best repeated no-fold/default-builder observation before rerun:

| Model | Avg ms | Median ms |
|---|---:|---:|
| SIAV2 P4small reinforced deploy | 1.5827 | 1.4485 |
| YOLOv7-W6 deploy | 3.9395 | 3.6856 |

## 6. TRT optimization experiments

The following were tested and rejected:

| Export/build setting | SIAV2 avg ms | W6 avg ms | Decision |
|---|---:|---:|---|
| No constant folding, default builder | 1.6241 | 3.9961 | Final generated setting |
| Constant folding + builder level 5 | 1.6691 | 3.8622 | Rejected: SIAV2 slower, layer names less parseable |
| No constant folding + builder level 5 | 1.7671 | 4.3381 | Rejected: slower |

Main SIAV2 bottlenecks in final profile:

- Detect/decode: `0.3374 ms`
- Backbone: `0.6904 ms`
- Neck: `0.5969 ms`
- Top layer index: `model.95` detect/decode aggregate, `0.3374 ms`

Possible future speed option:

- Raw-head export can remove/defer most detect decode work from the TensorRT engine.
- This would change the output contract from decoded `(1,25200,85)` to raw head tensors, so it was not applied to the final default.

## 7. Final decision

Use the P4/P5/P6 SIAV2 architecture as the speed target. Do not restore P3 in deploy.

Default training recipe:

- Use the previous relaxed P4-small strategy as default.
- Keep `hyp.siav2-p4small-aux-reinforced.yaml` as an optional experiment, not the default.

Final deploy/profile artifacts produced:

- `weights/siav2-p4small-aux-reinforced-coco128-best.pt`
- `weights/siav2-p4small-aux-reinforced-coco128-last.pt`
- `weights/siav2-p4small-aux-reinforced-coco128-deploy.pt`
- `weights/w6-coco128-e100-rerun-throttle4s-best.pt`
- `weights/w6-coco128-e100-rerun-throttle4s-last.pt`
- `weights/w6-coco128-rerun-throttle4s-deploy.pt`
- `runs/siav2/onnx/siav2-p4small-reinforced-coco80-1280.onnx`
- `runs/siav2/onnx/w6-coco80-rerun-1280.onnx`
- `runs/siav2/trt/siav2-p4small-reinforced-coco80-1280.fp16.engine`
- `runs/siav2/trt/w6-coco80-rerun-1280.fp16.engine`
- `runs/siav2/trt/reinforced_coco80_trt_summary.json`
- `runs/siav2/trt/reinforced_coco80_trt_summary.csv`

Note: this COCO128 run uses COCO-style `nc=80` heads for smoke comparison. The SIAV2 production `nc=16` head should be at least as light in the detect layer, but it still needs its own final export/profile after SIAV2 dataset training.
