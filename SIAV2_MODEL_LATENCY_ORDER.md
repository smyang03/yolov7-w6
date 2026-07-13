# SIAV2 Model Latency Order

Date: 2026-07-11

Measurement condition:

- TensorRT FP16
- Batch 1
- Random weights
- No training run
- Latency format: `avg / median ms`

Important separation:

- `YOLOv7-P5 640` comparison uses `cfg/deploy/yolov7-nc16.yaml`, derived from official `cfg/deploy/yolov7.yaml`. There is no official local `yolov7-l.yaml` in this repo.
- `W6 1280` comparison uses `cfg/deploy/yolov7-w6-nc16.yaml`.
- Do not compare 640 and 1280 as the same workload. They are listed separately.

## YOLOv7-P5 640 기준: Latency 오름차순

Baseline: `YOLOv7-P5 640`, `cfg/deploy/yolov7-nc16.yaml`, 3.1172 / 2.6022 ms.

| rank | model | base | resolution | what was changed | latency | speedup vs YOLOv7-P5 640 | status |
|---:|---|---|---:|---|---:|---:|---|
| 1 | `siav2-anchorfree-640` | YOLOv7-P5 640 기준 설계 후보 | 640 | SIAV2 backbone/neck + anchor-free proxy head. One prediction per grid cell. | 1.9239 / 1.7450 | 1.62x | Fastest, latency-only |
| 2 | `siav2-dual-anchorfree-640` | YOLOv7-P5 640 기준 설계 후보 | 640 | SIAV2 backbone/neck + anchor-based branch + anchor-free branch. | 1.9986 / 1.7764 | 1.56x | Latency-only |
| 3 | `siav2-final-640` | YOLOv7-P5 640 기준 설계 후보 | 640 | W6-derived SIAV2. P3 removed, P4/P5/P6 only, width 0.25, anchor head kept. | 2.0827 / 1.7859 | 1.50x | Train-ready cfg family |
| 4 | `siav2-dualhead-640` | YOLOv7-P5 640 기준 설계 후보 | 640 | SIAV2 backbone/neck + two anchor-based detection branches. | 2.1300 / 1.8129 | 1.46x | Latency-only |
| 5 | `baseline-yolov7-p5-640` | YOLOv7-P5 baseline | 640 | Official `yolov7.yaml` derived, `nc=16`, P3/P4/P5 Detect, depth 1.0, width 1.0. | 3.1172 / 2.6022 | 1.00x | Baseline |
| 6 | `baseline-w6-640` | W6 reference at 640 | 640 | W6 measured at 640 only as extra reference. Not the YOLOv7-P5 baseline. | 3.2888 / 2.8132 | 0.95x | Reference |

## W6 1280 기준: Latency 오름차순

Baseline: `YOLOv7-W6 1280`, `cfg/deploy/yolov7-w6-nc16.yaml`, 4.7009 / 4.4584 ms.

| rank | model | base | resolution | what was changed | latency | speedup vs W6 1280 | status |
|---:|---|---|---:|---|---:|---:|---|
| 1 | `siav2-anchorfree-1280` | W6 1280 기준 설계 후보 | 1280 | SIAV2 backbone/neck + anchor-free proxy head. One prediction per grid cell. | 2.0787 / 1.8218 | 2.26x | Fastest, latency-only |
| 2 | `siav2-final-1280` / `siav2-p4p6-pruned-w250-1280` | W6 1280 기준 설계 후보 | 1280 | W6 improved. P3 removed, P4/P5/P6 only, width 0.25, anchor head kept. | 2.1856 / 1.9552 | 2.15x | Final train-ready candidate |
| 3 | `siav2-dual-anchorfree-1280` | W6 1280 기준 설계 후보 | 1280 | SIAV2 backbone/neck + anchor-based branch + anchor-free branch. | 2.3912 / 2.0265 | 1.97x | Latency-only, avg below 2x |
| 4 | `siav2-p4p6-pruned-w300-1280` | W6 1280 기준 설계 후보 | 1280 | P3 removed, P4/P5/P6 only, width 0.30. | 2.4012 / 2.2189 | 1.96x | Near miss |
| 5 | `siav2-dualhead-1280` | W6 1280 기준 설계 후보 | 1280 | SIAV2 backbone/neck + two anchor-based detection branches. | 2.5697 / 2.3350 | 1.83x | Latency-only |
| 6 | `siav2-p4p6-pruned-w400-1280` | W6 1280 기준 설계 후보 | 1280 | P3 removed, P4/P5/P6 only, width 0.40. | 2.5868 / 2.3970 | 1.82x | Below 2x |
| 7 | `siav2-p4p6-pruned-w600-1280` | W6 1280 기준 설계 후보 | 1280 | P3 removed, P4/P5/P6 only, width 0.60. | 3.1009 / 2.9509 | 1.52x | Below 2x |
| 8 | `siav2-p4p6-pruned-w550-1280` | W6 1280 기준 설계 후보 | 1280 | P3 removed, P4/P5/P6 only, width 0.55. | 3.1152 / 2.9355 | 1.51x | Below 2x |
| 9 | `siav2-4scale-w500-1280` | W6 1280 기준 설계 후보 | 1280 | W6 4-scale structure kept, P3/P4/P5/P6 kept, width reduced to 0.50. | 3.1367 / 2.9608 | 1.50x | Below 2x |
| 10 | `baseline-w6-1280` | W6 baseline | 1280 | W6 4-scale, `nc=16`, P3/P4/P5/P6 Detect. | 4.7009 / 4.4584 | 1.00x | Baseline |

## Final Read

- YOLOv7-P5 640 기준 fastest: `siav2-anchorfree-640`, 1.9239 ms avg, but latency-only.
- YOLOv7-P5 640 기준 train-ready family: `siav2-final-640`, 2.0827 ms avg.
- W6 1280 기준 fastest: `siav2-anchorfree-1280`, 2.0787 ms avg, but latency-only.
- W6 1280 기준 final train-ready candidate: `siav2-final-1280` / `siav2-p4p6-pruned-w250-1280`, 2.1856 ms avg, 2.15x faster than W6 1280.
