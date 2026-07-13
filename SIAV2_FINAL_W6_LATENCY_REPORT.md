# SIAV2 Final W6 Training And Latency Report

Date: 2026-07-13

## 1. Scope

This report compares:

- Final selected SIAV2 training performance against the existing W6 COCO128 100epoch baseline.
- All TensorRT FP16 latency measurements currently available in `runs/siav2/trt`.

Important distinction:

- COCO128 training probes use `nc=80`.
- The earlier design latency table mostly uses `nc=16` random/deploy graphs.
- Do not directly mix `nc=80` trained latency and `nc=16` design latency as if they are the same benchmark.
- Older `runs/siav2/latency_*.csv` files are PyTorch/CFG timing probes, not the final TensorRT table, so they are not mixed into the TRT latency ranking.
- Low-risk raw-head / EfficientNMS-ready deploy profiling is tracked separately in `SIAV2_LOW_RISK_DEPLOY_REPORT.md`.
- P3-lite / deep-head tradeoff latency profiling is tracked separately in `SIAV2_P3_TRADEOFF_LATENCY_REPORT.md`.
- Code-safety fixes and P3-lite training cfg verification are tracked separately in `SIAV2_CODE_IMPROVEMENTS_REPORT.md`.
- W6-to-SIAV2 distillation and validation protocol are tracked separately in `SIAV2_DISTILLATION_VALIDATION_PLAN.md`.

## 2. Final training decision vs W6

Final selected training recipe is **SIAV2 P4small aux relaxed**, not reinforced.

Reason:

- `reinforced` beat W6, but it underperformed the earlier `relaxed` run.
- P3 compensation should therefore stay as relaxed P4-small anchor/assignment/balance.
- The extra small-object loss and dynamic-k reinforcement remains optional, not default.

COCO128 100epoch, image 1280, random-init smoke comparison:

| Model | Run | Final total loss | P | R | mAP50 | mAP50-95 | Read |
|---|---|---:|---:|---:|---:|---:|---|
| SIAV2 P4small aux relaxed | `siav2_p4small_aux_relaxed_1280_e100` | 0.2752 | 0.2880 | 0.008362 | 0.002400 | 0.0004867 | Final selected training recipe |
| SIAV2 P4small aux base | `siav2_p4small_aux_1280_e100` | 0.2680 | 0.5704 | 0.003352 | 0.001330 | 0.0003951 | Good, but below relaxed |
| SIAV2 P4small aux reinforced | `siav2_p4small_aux_reinforced_1280_e100` | 0.3227 | 0.8470 | 0.001083 | 0.0003228 | 0.0000768 | Optional, not default |
| SIAV2 P4small aux conservative | `siav2_p4small_aux_conservative_1280_e100` | 0.2520 | 0.2396 | 0.0002773 | 0.00002496 | 0.000004449 | Rejected |
| YOLOv7-W6 existing baseline | `w6_aux_1280_e100` | 0.2046 | 0.00002951 | 0.002773 | 0.000007673 | 0.000001252 | Existing W6 reference |
| YOLOv7-W6 rerun throttle4s | `w6_aux_1280_e100_rerun_throttle4s` | 0.2049 | 0.00003152 | 0.003050 | 0.000008152 | 0.000001254 | Recheck under GPU throttle |

Against the existing W6 baseline:

| Metric | SIAV2 relaxed | W6 existing | Ratio |
|---|---:|---:|---:|
| Final total loss | 0.2752 | 0.2046 | W6 lower loss |
| Precision | 0.2880 | 0.00002951 | SIAV2 much higher |
| Recall | 0.008362 | 0.002773 | SIAV2 3.02x |
| mAP50 | 0.002400 | 0.000007673 | SIAV2 312.8x |
| mAP50-95 | 0.0004867 | 0.000001252 | SIAV2 388.7x |

Interpretation:

- W6 has lower train loss, but SIAV2 relaxed has much better validation mAP in this tiny random-init COCO128 probe.
- Absolute mAP values are still very low, so this is a trainability/procedure check, not a production accuracy claim.
- Final training default remains `data/hyp.siav2-p4small-aux-relaxed.yaml`.

## 3. TRT FP16 latency: nc16 design models

Source: `runs/siav2/trt/today_all_speed_comparison.csv`

Batch 1, TensorRT FP16, random/deploy graph latency.

| Model | Type | Img | Avg ms | Median ms | Notes |
|---|---|---:|---:|---:|---|
| `siav2-anchorfree-640` | SIAV2 head experiment | 640 | 1.9239 | 1.7450 | Fastest 640 head experiment, latency-only |
| `siav2-dual-anchorfree-640` | SIAV2 head experiment | 640 | 1.9986 | 1.7764 | Dual + anchor-free, latency-only |
| `siav2-final-640` | SIAV2 final family | 640 | 2.0827 | 1.7859 | P3 removed family |
| `siav2-dualhead-640` | SIAV2 head experiment | 640 | 2.1300 | 1.8129 | Dual head cost visible |
| `baseline-yolov7-p5-640` | Baseline | 640 | 3.1172 | 2.6022 | `yolov7-nc16-640` P5 baseline |
| `baseline-w6-640` | Baseline | 640 | 3.2888 | 2.8132 | W6 graph at 640 |
| `siav2-anchorfree-1280` | SIAV2 head experiment | 1280 | 2.0787 | 1.8218 | Fastest 1280 latency-only head |
| `siav2-final-1280` | SIAV2 final family | 1280 | 2.1856 | 1.9552 | Same measured log as `p4p6-pruned-w250` |
| `siav2-p4p6-pruned-w250-1280` | SIAV2 width sweep | 1280 | 2.1856 | 1.9552 | 2x speed gate pass |
| `siav2-dual-anchorfree-1280` | SIAV2 head experiment | 1280 | 2.3912 | 2.0265 | Dual + anchor-free |
| `siav2-p4p6-pruned-w300-1280` | SIAV2 width sweep | 1280 | 2.4012 | 2.2189 | Median passes 2x gate, avg does not |
| `siav2-dualhead-1280` | SIAV2 head experiment | 1280 | 2.5697 | 2.3350 | Slower than final P4/P6 |
| `siav2-p4p6-pruned-w400-1280` | SIAV2 width sweep | 1280 | 2.5868 | 2.3970 | Too slow for 2x avg gate |
| `siav2-p4p6-pruned-w600-1280` | SIAV2 width sweep | 1280 | 3.1009 | 2.9509 | Too slow |
| `siav2-p4p6-pruned-w550-1280` | SIAV2 width sweep | 1280 | 3.1152 | 2.9355 | Too slow |
| `siav2-4scale-w500-1280` | SIAV2 4-scale experiment | 1280 | 3.1367 | 2.9608 | P3-like cost returns |
| `baseline-w6-1280` | Baseline | 1280 | 4.7009 | 4.4584 | Main W6 nc16 latency baseline |

Key nc16 latency conclusions:

- Best train-ready 1280 family remains `siav2-p4p6-pruned-w250-1280`: `2.1856 / 1.9552 ms`.
- Versus W6 1280 nc16: `2.15x` faster by avg and `2.28x` faster by median.
- 4-scale/P3-like variants lose too much speed.
- Anchor-free variants are faster in latency-only tests, but they are not selected until loss/assignment/training is implemented.

## 4. TRT FP16 latency: trained COCO80 deploy profiles

Source: `runs/siav2/trt/reinforced_coco80_trt_summary.csv`

These use trained COCO128 checkpoints converted from aux to deploy. Because the smoke dataset is COCO128, the heads are `nc=80`.

| Model | Deploy heads | Output | Avg ms | Median ms | Backbone ms | Neck ms | Detect/decode ms | Avg speedup vs W6 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SIAV2 P4small reinforced COCO80 | P4/P5/P6 | decoded `(1,25200,85)` | 1.6241 | 1.4601 | 0.6904 | 0.5969 | 0.3374 | 2.46x |
| YOLOv7-W6 COCO80 rerun | P3/P4/P5/P6 | decoded `(1,102000,85)` | 3.9961 | 3.7349 | 1.7390 | 1.6712 | 0.5321 | 1.00x |

Best repeated no-fold/default-builder observation:

| Model | Avg ms | Median ms |
|---|---:|---:|
| SIAV2 P4small reinforced COCO80 | 1.5827 | 1.4485 |
| YOLOv7-W6 COCO80 rerun | 3.9395 | 3.6856 |

Rejected TRT build/export variants:

| Variant | SIAV2 avg ms | W6 avg ms | Decision |
|---|---:|---:|---|
| Constant folding + builder level 5 | 1.6691 | 3.8622 | Rejected because SIAV2 slowed |
| No constant folding + builder level 5 | 1.7671 | 4.3381 | Rejected because both slowed |

## 5. Final selection

Training:

- Select `SIAV2 P4small aux relaxed`.
- Use `data/hyp.siav2-p4small-aux-relaxed.yaml` as the default P3-removal compensation recipe.
- Keep `data/hyp.siav2-p4small-aux-reinforced.yaml` only as an optional experiment.

Architecture/deploy:

- Keep P3 removed.
- Use P4/P5/P6 deploy family.
- Do not select 4-scale/P3-like variants because they weaken the speed target.

Latency:

- nc16 design reference: `siav2-p4p6-pruned-w250-1280`, `2.1856 / 1.9552 ms`.
- trained COCO80 deploy reference: SIAV2 P4small converted deploy, `1.6241 / 1.4601 ms`.

Main caveat:

- COCO128 random-init 100epoch is only a smoke/training-path comparison. The actual SIAV2 `nc=16` full-data training must be run before making a production accuracy claim.
