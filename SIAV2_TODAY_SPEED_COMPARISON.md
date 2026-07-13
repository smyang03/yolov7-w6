# SIAV2 Today Speed Comparison

Date: 2026-07-11

## Scope

- No training was run.
- Measurements are TensorRT FP16, batch 1, random weights.
- GPU used for these measurements: RTX 4090.
- `640` comparison target: YOLOv7-P5 640 reference, implemented as `cfg/deploy/yolov7-nc16.yaml`.
- `cfg/deploy/yolov7-nc16.yaml` is derived from official `cfg/deploy/yolov7.yaml` with `nc=16`, `depth_multiple=1.0`, `width_multiple=1.0`, and P3/P4/P5 Detect. It is not W6 and not `yolov7x`.
- `1280` comparison target: YOLOv7-W6 deploy cfg with `nc=16`.
- Anchor-free heads here are latency-only ablations. Loss, target assignment, and mAP validity are not implemented yet.

Raw comparison CSV:

- `runs/siav2/trt/today_all_speed_comparison.csv`
- `runs/siav2/trt/head_ablation_latency.csv`
- `runs/siav2/trt/head_ablation_weights.csv`

Latency-ordered model list:

- `SIAV2_MODEL_LATENCY_ORDER.md`

## Correct Comparison Sets

There are two separate comparisons. Do not mix the 640 baseline with the 1280 baseline.

| set | correct baseline | resolution | baseline cfg | correct meaning | baseline speed |
|---|---|---:|---|---|---:|
| A | YOLOv7-P5 640 reference | 640 | `cfg/deploy/yolov7-nc16.yaml` | Official `yolov7.yaml` derived model, `nc=16`, P3/P4/P5 Detect. This is the 640 P5 reference. It is not W6 and not an official `yolov7-l.yaml`. | 3.1172 / 2.6022 ms |
| B | YOLOv7-W6/1280 baseline | 1280 | `cfg/deploy/yolov7-w6-nc16.yaml` | W6 4-scale model, `nc=16`, P3/P4/P5/P6 Detect. This is the real 1280 baseline. | 4.7009 / 4.4584 ms |

## Correct 640 Reading

For 640, read the table like this:

| role | model | what it is | speed |
|---|---|---|---:|
| correct 640 baseline | `baseline-yolov7-p5-640` | YOLOv7-P5 640 reference, `yolov7.yaml` derived, `nc=16` | 3.1172 / 2.6022 ms |
| fastest 640 latency ablation | `siav2-anchorfree-640` | SIAV2 backbone/neck plus anchor-free proxy head | 1.9239 / 1.7450 ms |
| second fastest 640 latency ablation | `siav2-dual-anchorfree-640` | SIAV2 backbone/neck plus anchor branch and anchor-free branch together | 1.9986 / 1.7764 ms |
| train-ready SIAV2 family at 640 | `siav2-final-640` | W6-derived lightweight SIAV2, P3 removed, P4/P5/P6 only, width 0.25 | 2.0827 / 1.7859 ms |
| slower head ablation | `siav2-dualhead-640` | SIAV2 backbone/neck plus two anchor-based heads | 2.1300 / 1.8129 ms |

So the clean 640 answer is:

- Baseline to quote: `YOLOv7-P5 640 reference`, 3.1172 ms avg.
- Fastest measured 640 model: `siav2-anchorfree-640`, 1.9239 ms avg, but latency-only.
- Correct train-ready SIAV2-family 640 model: `siav2-final-640`, 2.0827 ms avg.

## Correct 1280 Reading

For 1280, read the table like this:

| role | model | what it is | speed |
|---|---|---|---:|
| correct 1280 baseline | `baseline-w6-1280` | YOLOv7-W6/1280, `nc=16`, P3/P4/P5/P6 Detect | 4.7009 / 4.4584 ms |
| fastest 1280 latency ablation | `siav2-anchorfree-1280` | SIAV2 backbone/neck plus anchor-free proxy head | 2.0787 / 1.8218 ms |
| final train-ready 1280 candidate | `siav2-final-1280` / `siav2-p4p6-pruned-w250-1280` | W6 improved: P3 removed, P4/P5/P6 only, width 0.25, anchor head kept | 2.1856 / 1.9552 ms |
| combined head ablation | `siav2-dual-anchorfree-1280` | Anchor branch plus anchor-free branch together | 2.3912 / 2.0265 ms |
| near miss width trial | `siav2-p4p6-pruned-w300-1280` | Same as final but width 0.30 | 2.4012 / 2.2189 ms |
| slower head ablation | `siav2-dualhead-1280` | Two anchor-based heads | 2.5697 / 2.3350 ms |

So the clean 1280 answer is:

- Baseline to quote: `YOLOv7-W6/1280`, 4.7009 ms avg.
- Fastest measured 1280 model: `siav2-anchorfree-1280`, 2.0787 ms avg, but latency-only.
- Correct final train-ready 1280 model: `siav2-final-1280`, 2.1856 ms avg, 2.15x faster than W6/1280.

## Change Labels

| label | actual change |
|---|---|
| `4scale-w500` | W6 4-scale structure kept; only width reduced to 0.50. P3 still exists. |
| `p4p6-pruned-wX` | P3/8 removed; only P4/P5/P6 kept; width set to X. |
| `final` / `p4p6-pruned-w250` | Final train-ready SIAV2 candidate: P3 removed, P4/P5/P6 only, width 0.25, anchor head kept. |
| `dualhead` | Same SIAV2 backbone/neck, but two anchor-based detection branches. Latency ablation. |
| `anchorfree` | Same SIAV2 backbone/neck, but anchor-free proxy head. Fastest, latency-only. |
| `dual-anchorfree` | Anchor-based branch and anchor-free branch together. Latency ablation. |

## Full Detailed Speed Table

Latency is `avg / median`. Speedup is calculated against the target baseline for the same resolution: `YOLOv7-P5-640` for 640 rows, and `YOLOv7-W6-1280` for 1280 rows.

| baseline | resolution | model | cfg / weight basis | what changed | latency ms | speedup | status |
|---|---:|---|---|---|---:|---:|---|
| YOLOv7-P5 | 640 | `baseline-yolov7-p5-640` | `cfg/deploy/yolov7-nc16.yaml`, derived from `yolov7.yaml` | Baseline: `nc=16`, P3/P4/P5 Detect, depth 1.0, width 1.0. This is the 640 P5 reference, not W6 and not `yolov7x`. | 3.1172 / 2.6022 | 1.00x | 640 baseline |
| YOLOv7-P5 | 640 | `baseline-w6-640` | `cfg/deploy/yolov7-w6-nc16.yaml` | W6 baseline measured at 640 for reference only. | 3.2888 / 2.8132 | 0.95x | reference |
| YOLOv7-P5 | 640 | `siav2-anchorfree-640` | `cfg/deploy/yolov7-l6-siav2-anchorfree.yaml` | Final SIAV2 backbone/neck plus `AFDetect`; anchor-free latency proxy, one prediction per grid cell. | 1.9239 / 1.7450 | 1.62x | fastest 640, latency-only |
| YOLOv7-P5 | 640 | `siav2-dual-anchorfree-640` | `cfg/deploy/yolov7-l6-siav2-dual_anchorfree.yaml` | Final SIAV2 backbone/neck plus anchor-based branch and anchor-free branch together. | 1.9986 / 1.7764 | 1.56x | latency-only |
| YOLOv7-P5 | 640 | `siav2-final-640` | `cfg/deploy/yolov7-l6-siav2.yaml` | W6-derived lightweight model, width 0.25, P3/8 removed, P4/P5/P6 only, 3-scale anchor head, stride 16/32/64. | 2.0827 / 1.7859 | 1.50x | train-ready cfg family |
| YOLOv7-P5 | 640 | `siav2-dualhead-640` | `cfg/deploy/yolov7-l6-siav2-dualhead.yaml` | Final SIAV2 backbone/neck plus `DualDetect`, two anchor-based detection branches. | 2.1300 / 1.8129 | 1.46x | latency-only |
| YOLOv7-W6 | 1280 | `baseline-w6-1280` | `cfg/deploy/yolov7-w6-nc16.yaml` | Baseline: W6 4-scale, `nc=16`, P3/P4/P5/P6 Detect. | 4.7009 / 4.4584 | 1.00x | 1280 baseline |
| YOLOv7-W6 | 1280 | `siav2-anchorfree-1280` | `cfg/deploy/yolov7-l6-siav2-anchorfree.yaml` | Final SIAV2 backbone/neck plus `AFDetect`; fastest anchor-free latency proxy. | 2.0787 / 1.8218 | 2.26x | fastest 1280, latency-only |
| YOLOv7-W6 | 1280 | `siav2-final-1280` / `siav2-p4p6-pruned-w250-1280` | `cfg/deploy/yolov7-l6-siav2.yaml`, `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w250.yaml` | P3/8 removed, P4/P5/P6 only, width 0.25, 3-scale anchor head. | 2.1856 / 1.9552 | 2.15x | final train-ready candidate |
| YOLOv7-W6 | 1280 | `siav2-dual-anchorfree-1280` | `cfg/deploy/yolov7-l6-siav2-dual_anchorfree.yaml` | Final SIAV2 backbone/neck plus anchor-based branch and anchor-free branch together. | 2.3912 / 2.0265 | 1.97x | latency-only, avg below 2x |
| YOLOv7-W6 | 1280 | `siav2-p4p6-pruned-w300-1280` | `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w300.yaml` | P3/8 removed, P4/P5/P6 only, width 0.30. | 2.4012 / 2.2189 | 1.96x | avg below 2x |
| YOLOv7-W6 | 1280 | `siav2-dualhead-1280` | `cfg/deploy/yolov7-l6-siav2-dualhead.yaml` | Final SIAV2 backbone/neck plus `DualDetect`, two anchor-based detection branches. | 2.5697 / 2.3350 | 1.83x | latency-only |
| YOLOv7-W6 | 1280 | `siav2-p4p6-pruned-w400-1280` | `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w400.yaml` | P3/8 removed, P4/P5/P6 only, width 0.40. | 2.5868 / 2.3970 | 1.82x | below 2x |
| YOLOv7-W6 | 1280 | `siav2-p4p6-pruned-w600-1280` | `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w600.yaml` | P3/8 removed, P4/P5/P6 only, width 0.60. | 3.1009 / 2.9509 | 1.52x | below 2x |
| YOLOv7-W6 | 1280 | `siav2-p4p6-pruned-w550-1280` | `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w550.yaml` | P3/8 removed, P4/P5/P6 only, width 0.55. | 3.1152 / 2.9355 | 1.51x | below 2x |
| YOLOv7-W6 | 1280 | `siav2-4scale-w500-1280` | `cfg/deploy/yolov7-l6-siav2-w500.yaml` | W6 4-scale structure kept, width reduced to 0.50, P3/P4/P5/P6 kept. | 3.1367 / 2.9608 | 1.50x | below 2x |

## 640 Speed

Speedup is calculated against `baseline-yolov7-p5-640` avg latency. This baseline is the YOLOv7-P5 640 reference in this report: `cfg/deploy/yolov7-nc16.yaml`, derived from official `cfg/deploy/yolov7.yaml`.

| rank | model | type | avg ms | median ms | speedup |
|---:|---|---|---:|---:|---:|
| 1 | `siav2-anchorfree-640` | anchor-free head | 1.9239 | 1.7450 | 1.62x |
| 2 | `siav2-dual-anchorfree-640` | dual + anchor-free | 1.9986 | 1.7764 | 1.56x |
| 3 | `siav2-final-640` | final W6-improved deploy | 2.0827 | 1.7859 | 1.50x |
| 4 | `siav2-dualhead-640` | dual anchor head | 2.1300 | 1.8129 | 1.46x |
| 5 | `baseline-yolov7-p5-640` | YOLOv7-P5 640 reference, `yolov7.yaml` derived | 3.1172 | 2.6022 | 1.00x |
| 6 | `baseline-w6-640` | YOLOv7-W6 baseline | 3.2888 | 2.8132 | 0.95x |

## 1280 Speed

Speedup is calculated against `baseline-w6-1280` avg latency.

| rank | model | type | avg ms | median ms | speedup |
|---:|---|---|---:|---:|---:|
| 1 | `siav2-anchorfree-1280` | anchor-free head | 2.0787 | 1.8218 | 2.26x |
| 2 | `siav2-final-1280` / `siav2-p4p6-pruned-w250-1280` | final W6-improved deploy | 2.1856 | 1.9552 | 2.15x |
| 3 | `siav2-dual-anchorfree-1280` | dual + anchor-free | 2.3912 | 2.0265 | 1.97x |
| 4 | `siav2-p4p6-pruned-w300-1280` | P4/P5/P6 pruned width 0.30 | 2.4012 | 2.2189 | 1.96x |
| 5 | `siav2-dualhead-1280` | dual anchor head | 2.5697 | 2.3350 | 1.83x |
| 6 | `siav2-p4p6-pruned-w400-1280` | P4/P5/P6 pruned width 0.40 | 2.5868 | 2.3970 | 1.82x |
| 7 | `siav2-p4p6-pruned-w600-1280` | P4/P5/P6 pruned width 0.60 | 3.1009 | 2.9509 | 1.52x |
| 8 | `siav2-p4p6-pruned-w550-1280` | P4/P5/P6 pruned width 0.55 | 3.1152 | 2.9355 | 1.51x |
| 9 | `siav2-4scale-w500-1280` | 4-scale width 0.50 | 3.1367 | 2.9608 | 1.50x |
| 10 | `baseline-w6-1280` | YOLOv7-W6 baseline | 4.7009 | 4.4584 | 1.00x |

## Weight Files

| model | weight | size |
|---|---|---:|
| W6 deploy baseline | `weights/random-yolov7-w6-nc16-deploy.pt` | 267.29 MB |
| SIAV2 final deploy | `weights/random-yolov7-l6-siav2-deploy.pt` | 16.57 MB |
| SIAV2 training cfg random init | `weights/random-yolov7-l6-siav2.pt` | 19.31 MB |
| SIAV2 final P4/P5/P6 pruned w250 | `weights/random-yolov7-l6-siav2-p4p6-pruned-w250.pt` | 19.32 MB |
| SIAV2 dual head | `weights/random-yolov7-l6-siav2-dualhead.pt` | 16.71 MB |
| SIAV2 anchor-free | `weights/random-yolov7-l6-siav2-anchorfree.pt` | 16.48 MB |
| SIAV2 dual + anchor-free | `weights/random-yolov7-l6-siav2-dual_anchorfree.pt` | 16.62 MB |

## Decision

The final pre-training model remains `siav2-final-1280`, equivalent to `siav2-p4p6-pruned-w250-1280` for TRT latency. It clears the 2x speed target against W6 1280 by avg latency and median latency:

- W6 1280 baseline: 4.7009 ms avg, 4.4584 ms median.
- SIAV2 final 1280: 2.1856 ms avg, 1.9552 ms median.
- Avg speedup: 2.15x.
- Median speedup: 2.28x.

Anchor-free is the fastest latency ablation, but it is not a train-ready final candidate yet because the training loss/assignment path is not implemented. Dual + anchor-free improves over dual-head, but misses the 2x avg target slightly at 1280.
