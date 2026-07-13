# SIAV2 P3 Tradeoff TensorRT Latency Report

Date: 2026-07-13

## Scope

This pass measures latency only. No training script was run.

Goal: check whether P3 can be restored in a very light form, or whether removing a deeper head can keep latency near the current selected SIAV2 model.

Measured at:

- Input: `1280`
- Classes: `nc=16`
- Batch: `1`
- Precision: TensorRT FP16
- Data transfers: disabled in `trtexec`

## Candidates

| Candidate | Detect Scales | Change |
|---|---|---|
| `siav2-p4p6-pruned-w250` | P4/P5/P6 | Current selected latency baseline, P3 removed |
| `siav2-p4p5-pruned-w250` | P4/P5 | Remove deep P6 from current pruned path |
| `siav2-p3lite-p4p6-w250` | P3-lite/P4/P5/P6 | Add a side P3-lite branch, keep P6 |
| `siav2-p3lite-p4p5-w250` | P3-lite/P4/P5 | Add a side P3-lite branch, remove P6 |
| `siav2-p3full-p4p5-w250` | P3/P4/P5 | Restore original W6-style P3 neck, remove P6 |

P3-lite branch:

```text
P4 feature -> 1x1 reduce -> upsample
backbone P3 -> 1x1 route
concat -> 1x1 -> 3x3 -> P3 detect feature
```

This branch does not feed back into P4/P5/P6. It is only a lightweight P3 detection side branch.

## Output Size

| Candidate | Decoded Output Shape | Raw Head Shapes |
|---|---:|---|
| P4/P5/P6 baseline | `(1, 25200, 21)` | `80x80`, `40x40`, `20x20` |
| P4/P5 only | `(1, 24000, 21)` | `80x80`, `40x40` |
| P3-lite/P4/P5/P6 | `(1, 102000, 21)` | `160x160`, `80x80`, `40x40`, `20x20` |
| P3-lite/P4/P5 | `(1, 100800, 21)` | `160x160`, `80x80`, `40x40` |
| P3-full/P4/P5 | `(1, 100800, 21)` | `160x160`, `80x80`, `40x40` |

P3 adds many decoded candidates. Therefore decoded latency includes a large decode/concat penalty that should be separated from pure feature/head compute when planning EfficientNMS.

## Decoded TensorRT FP16 Latency

Source: `runs/siav2/trt/p3_tradeoff_trt_summary.csv`

| Candidate | Avg ms | Median ms | Backbone ms | Neck ms | Detect/Decode ms | Delta vs Current Avg |
|---|---:|---:|---:|---:|---:|---:|
| P4/P5 only | 2.0022 | 1.7811 | 0.8170 | 0.6296 | 0.1922 | -3.82% |
| P4/P5/P6 current | 2.0817 | 1.9368 | 0.7673 | 0.7019 | 0.2440 | baseline |
| P3-lite/P4/P5 | 2.1402 | 1.8955 | 0.7996 | 0.6264 | 0.3391 | +2.81% |
| P3-lite/P4/P5/P6 | 2.2950 | 2.1292 | 0.7541 | 0.7411 | 0.3862 | +10.25% |
| P3-full/P4/P5 | 2.3917 | 2.2170 | 0.7821 | 0.8955 | 0.2903 | +14.89% |

Decoded result:

- Removing P6 only is fastest, but it does not solve the P3 removal risk.
- P3-lite plus P4/P5 is close to the current model: only `+2.81%` avg, and median is `2.13%` faster.
- P3-lite while keeping P6 costs about `+10.25%`.
- Full P3 neck plus P4/P5 is too expensive for this speed target.

## Raw-Head TensorRT FP16 Latency

Source: `runs/siav2/trt/p3_tradeoff_rawhead_trt_summary.csv`

| Candidate | Avg ms | Median ms | Backbone ms | Neck ms | Detect Head ms | Delta vs Current Raw Avg |
|---|---:|---:|---:|---:|---:|---:|
| P4/P5 only | 1.7599 | 1.6118 | 0.7573 | 0.6189 | 0.0219 | -14.68% |
| P3-lite/P4/P5 | 1.9976 | 1.7089 | 0.8337 | 0.7435 | 0.0415 | -3.15% |
| P4/P5/P6 current | 2.0626 | 1.8195 | 0.8452 | 0.7300 | 0.0868 | baseline |
| P3-lite/P4/P5/P6 | 2.2043 | 1.9325 | 0.7879 | 0.9138 | 0.0500 | +6.87% |
| P3-full/P4/P5 | 2.4697 | 2.1288 | 0.8430 | 1.0523 | 0.0804 | +19.74% |

Raw-head result:

- With decoded output removed, `P3-lite/P4/P5` becomes faster than the current raw baseline.
- `P3-lite/P4/P5` is the best P3-restoring latency tradeoff.
- `P3-full/P4/P5` is rejected on latency.

## Copy/Reformat

Decoded source: `runs/siav2/trt/p3_tradeoff_reformat_summary.csv`

| Candidate | Copy/Reformat ms | Copy/Reformat % |
|---|---:|---:|
| P4/P5/P6 current | 0.3697 | 8.880 |
| P4/P5 only | 0.4844 | 12.097 |
| P3-lite/P4/P5/P6 | 0.5455 | 11.888 |
| P3-lite/P4/P5 | 0.4454 | 10.407 |
| P3-full/P4/P5 | 0.5653 | 11.818 |

Raw-head source: `runs/siav2/trt/p3_tradeoff_rawhead_reformat_summary.csv`

| Candidate | Copy/Reformat ms | Copy/Reformat % |
|---|---:|---:|
| P4/P5/P6 current | 0.3096 | 7.506 |
| P4/P5 only | 0.4343 | 12.341 |
| P3-lite/P4/P5/P6 | 0.5139 | 11.657 |
| P3-lite/P4/P5 | 0.4531 | 11.341 |
| P3-full/P4/P5 | 0.5994 | 12.135 |

P3 variants increase copy/reformat ratio. The main practical cost is still the high-resolution P3 path and decoded candidate count.

## Decision

Best latency-only option:

- `siav2-p4p5-pruned-w250`
- Fastest, but it removes P6 and still has no P3. It is not a P3 compensation strategy.

Best P3 compensation option:

- `siav2-p3lite-p4p5-w250`
- Restores P3 at stride 8.
- Removes P6 to pay for the P3-lite branch.
- Decoded latency stays close to current selected model.
- Raw-head latency is faster than current selected raw-head baseline.

Rejected for now:

- `siav2-p3lite-p4p6-w250`: useful but costs too much when P6 is kept.
- `siav2-p3full-p4p5-w250`: P3 recovery is stronger, but latency cost is too high.

Recommendation:

- Keep `siav2-p4p6-pruned-w250` as the current speed-first selected model.
- Promote `siav2-p3lite-p4p5-w250` as the next training candidate if P3 removal risk is considered too high.
- Do not use full P3 neck unless accuracy testing proves P3-lite is insufficient.

## Artifacts

Created deploy cfgs:

- `cfg/deploy/yolov7-l6-siav2-p4p5-pruned-w250.yaml`
- `cfg/deploy/yolov7-l6-siav2-p3lite-p4p6-w250.yaml`
- `cfg/deploy/yolov7-l6-siav2-p3lite-p4p5-w250.yaml`
- `cfg/deploy/yolov7-l6-siav2-p3full-p4p5-w250.yaml`

Created training cfg:

- `cfg/training/yolov7-l6-siav2-p3lite-p4p5-w250.yaml`

Scripts:

- `tools/make_siav2_p3_tradeoff_variants.py`
- `scripts/profile_siav2_p3_tradeoff_trt.ps1`
- `scripts/profile_siav2_p3_tradeoff_rawhead_trt.ps1`

Generated summaries:

- `runs/siav2/trt/p3_tradeoff_trt_summary.csv`
- `runs/siav2/trt/p3_tradeoff_rawhead_trt_summary.csv`
- `runs/siav2/trt/p3_tradeoff_reformat_summary.csv`
- `runs/siav2/trt/p3_tradeoff_rawhead_reformat_summary.csv`
