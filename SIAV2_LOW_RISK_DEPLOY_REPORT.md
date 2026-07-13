# SIAV2 Low-Risk Deploy Preparation Report

Date: 2026-07-13

## Scope

This pass only includes low-risk work that does not change the train-time model architecture:

- Added raw-head ONNX export path for TensorRT / EfficientNMS preparation.
- Re-profiled nc16 TensorRT FP16 engines for decoded vs raw-head exports.
- Added layer-profile copy/reformat analysis.
- Documented the EfficientNMS-ready deployment structure.

Held for later review:

- Neck P4/P5 channel reduction.
- P4 head channel reduction.
- SPP/SPPCSPC replacement.
- Backbone width reduction.

## Code Changes

### `tools/export_trt_onnx.py`

Added `--raw-head`.

- Decoded export keeps the current YOLOv7 deploy output contract.
- Raw-head export returns per-scale raw Detect outputs instead of the decoded/concatenated tensor.
- NMS/end2end flags are disabled for raw-head export, so EfficientNMS can be attached as a separate deploy step.

Observed output shapes at `1280` / `nc16`:

- Decoded SIAV2: `(1, 25200, 21)`
- Raw SIAV2:
  - `(1, 3, 80, 80, 21)`
  - `(1, 3, 40, 40, 21)`
  - `(1, 3, 20, 20, 21)`

### `scripts/profile_siav2_nc16_rawhead_trt.ps1`

Added a repeatable export/build/profile script for:

- `siav2-p4p6-pruned-w250-nc16-decoded-1280`
- `siav2-p4p6-pruned-w250-nc16-raw-1280`
- `w6-nc16-decoded-1280`
- `w6-nc16-raw-1280`

All TensorRT engines were built with FP16.

### `tools/analyze_trt_reformat.py`

Added a TensorRT profile parser that summarizes copy/reformat rows from `trtexec --dumpProfile` logs.

## TRT FP16 Latency

Source: `runs/siav2/trt/nc16_rawhead_trt_summary.csv`

| Model | Output | Input | Avg ms | Median ms | Backbone ms | Neck ms | Detect/Decode ms | Other ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIAV2 P4/P6 pruned w250 nc16 | decoded | 1280 | 2.0776 | 1.9466 | 0.7923 | 0.6628 | 0.2588 | 0.3631 |
| SIAV2 P4/P6 pruned w250 nc16 | raw-head | 1280 | 2.1157 | 1.8512 | 0.8458 | 0.7645 | 0.1236 | 0.3822 |
| YOLOv7-W6 nc16 | decoded | 1280 | 4.8854 | 4.6875 | 2.0166 | 1.9081 | 0.4418 | 0.5191 |
| YOLOv7-W6 nc16 | raw-head | 1280 | 4.3662 | 4.1520 | 1.9635 | 1.8755 | 0.0563 | 0.4712 |

## Raw-Head Result

SIAV2:

- Detect/decode section improved from `0.2588 ms` to `0.1236 ms`.
- Median latency improved from `1.9466 ms` to `1.8512 ms` (`~4.9%` faster).
- Average latency changed from `2.0776 ms` to `2.1157 ms` (`~1.8%` slower).

Decision: do not replace the default SIAV2 decoded engine only from this engine-internal average result. Keep decoded export as the compatibility baseline, and keep raw-head export as the EfficientNMS-ready path.

W6:

- Detect/decode section improved from `0.4418 ms` to `0.0563 ms`.
- Average latency improved from `4.8854 ms` to `4.3662 ms` (`~10.6%` faster).
- Median latency improved from `4.6875 ms` to `4.1520 ms` (`~11.4%` faster).

Decision: raw-head is clearly useful for W6-style decoded-output removal.

## Copy/Reformat Summary

Source: `runs/siav2/trt/nc16_rawhead_reformat_summary.csv`

The `total_profile_avg_ms` column is the sum of TensorRT profile rows and is not the same accounting basis as end-to-end engine latency. Use this table for copy/reformat ratio only.

| Model | Output | Layer Rows | Copy/Reformat Rows | Profile Sum ms | Copy/Reformat ms | Copy/Reformat % |
|---|---:|---:|---:|---:|---:|---:|
| SIAV2 P4/P6 pruned w250 nc16 | decoded | 272 | 68 | 4.1546 | 0.3797 | 9.139 |
| SIAV2 P4/P6 pruned w250 nc16 | raw-head | 236 | 56 | 4.2318 | 0.2979 | 7.040 |
| YOLOv7-W6 nc16 | decoded | 339 | 87 | 9.7710 | 0.8598 | 8.800 |
| YOLOv7-W6 nc16 | raw-head | 291 | 72 | 8.7327 | 0.7388 | 8.460 |

Raw-head reduced copy/reformat work:

- SIAV2: `0.3797 ms` to `0.2979 ms` (`~21.5%` reduction).
- W6: `0.8598 ms` to `0.7388 ms` (`~14.1%` reduction).

Main observed copy/reformat areas:

- SiLU activation outputs around convolution blocks.
- Resize/concat boundaries in neck routing.
- Detect output cast/reshape/concat rows in decoded export.

The low-risk export path reduces Detect-side copy/reformat rows. Further neck/backbone copy reduction likely needs architectural layout or channel-routing changes, so it remains on hold.

## EfficientNMS-Ready Deploy Structure

Current decoded export:

```text
backbone/neck/head -> YOLO decode + concat -> decoded boxes/classes output
```

Raw-head export:

```text
backbone/neck/head -> raw P4/P5/P6 tensors -> decode step -> EfficientNMS_TRT -> final detections
```

Important deployment note:

- TensorRT EfficientNMS normally consumes decoded boxes and class scores.
- Therefore the raw head still needs a decode step before EfficientNMS.
- The decode step can be left in ONNX, implemented as a TensorRT plugin, or fused into a custom decode+NMS plugin.
- The current work prepares the raw output contract and profiling path, but does not yet attach EfficientNMS.

## Current Recommendation

For training preparation:

- Keep training architecture as `siav2-p4p6-pruned-w250-1280`.
- Do not apply neck/SPP/backbone/head-risk changes before first training.

For deployment preparation:

- Keep decoded ONNX/TRT as the validated compatibility path.
- Keep raw-head ONNX/TRT as the EfficientNMS-ready path.
- Next deploy-specific test should be `raw-head + decode + EfficientNMS_TRT`, measured including postprocess/data-transfer boundary if that is part of the real application latency.

## Verification

Completed:

- ONNX export smoke test for decoded and raw-head outputs.
- TensorRT FP16 engine build for all four nc16 variants.
- TensorRT FP16 latency profiling.
- TensorRT layer profile copy/reformat analysis.

No training script was run.
