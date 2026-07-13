# SIAV2 Full Model Comparison

Date: 2026-07-11

## Measurement Rules

- No training was run.
- Latency is TensorRT FP16, batch 1, random weights.
- Latency format is `avg / median ms`.
- `640` speedup uses `baseline-yolov7-p5-640` as the base.
- `1280` speedup uses `baseline-w6-1280` as the base.
- `YOLOv7-P5 640` means `cfg/deploy/yolov7-nc16.yaml`, derived from official `cfg/deploy/yolov7.yaml`. This repo does not have an official `yolov7-l.yaml`.

## Baseline Definition

| baseline id | correct name | resolution | cfg | structure | latency | role |
|---|---|---:|---|---|---:|---|
| `baseline-yolov7-p5-640` | YOLOv7-P5 640 | 640 | `cfg/deploy/yolov7-nc16.yaml` | `yolov7.yaml` derived, `nc=16`, P3/P4/P5 Detect, stride 8/16/32 | 3.1172 / 2.6022 | 640 comparison base |
| `baseline-w6-640` | YOLOv7-W6 640 | 640 | `cfg/deploy/yolov7-w6-nc16.yaml` | W6, `nc=16`, P3/P4/P5/P6 Detect, measured at 640 only as reference | 3.2888 / 2.8132 | reference only |
| `baseline-w6-1280` | YOLOv7-W6 1280 | 1280 | `cfg/deploy/yolov7-w6-nc16.yaml` | W6, `nc=16`, P3/P4/P5/P6 Detect | 4.7009 / 4.4584 | 1280 comparison base |

## Full Latency Comparison

Sorted inside each resolution by average latency.

| group | rank | model id | resolution | cfg | design change | latency | speedup | train-ready? | decision |
|---|---:|---|---:|---|---|---:|---:|---|---|
| 640 / YOLOv7-P5 base | 1 | `siav2-anchorfree-640` | 640 | `cfg/deploy/yolov7-l6-siav2-anchorfree.yaml` | SIAV2 backbone/neck plus `AFDetect`; anchor-free proxy, one prediction per grid cell | 1.9239 / 1.7450 | 1.62x | No | Fastest 640 latency ablation |
| 640 / YOLOv7-P5 base | 2 | `siav2-dual-anchorfree-640` | 640 | `cfg/deploy/yolov7-l6-siav2-dual_anchorfree.yaml` | SIAV2 backbone/neck plus anchor-based branch and anchor-free branch together | 1.9986 / 1.7764 | 1.56x | No | Latency-only dual/AF test |
| 640 / YOLOv7-P5 base | 3 | `siav2-final-640` | 640 | `cfg/deploy/yolov7-l6-siav2.yaml` | W6-derived SIAV2, P3 removed, P4/P5/P6 only, width 0.25, anchor head kept | 2.0827 / 1.7859 | 1.50x | Yes | Train-ready SIAV2 family at 640 |
| 640 / YOLOv7-P5 base | 4 | `siav2-dualhead-640` | 640 | `cfg/deploy/yolov7-l6-siav2-dualhead.yaml` | SIAV2 backbone/neck plus `DualDetect`, two anchor-based detection branches | 2.1300 / 1.8129 | 1.46x | No | Slower than AF variants |
| 640 / YOLOv7-P5 base | 5 | `baseline-yolov7-p5-640` | 640 | `cfg/deploy/yolov7-nc16.yaml` | Baseline, official `yolov7.yaml` derived, `nc=16` | 3.1172 / 2.6022 | 1.00x | Yes | 640 base |
| 640 / reference | 6 | `baseline-w6-640` | 640 | `cfg/deploy/yolov7-w6-nc16.yaml` | W6 measured at 640 for reference | 3.2888 / 2.8132 | 0.95x vs P5 640 | Yes | Not the 640 base |
| 1280 / W6 base | 1 | `siav2-anchorfree-1280` | 1280 | `cfg/deploy/yolov7-l6-siav2-anchorfree.yaml` | SIAV2 backbone/neck plus `AFDetect`; anchor-free proxy, one prediction per grid cell | 2.0787 / 1.8218 | 2.26x | No | Fastest 1280 latency ablation |
| 1280 / W6 base | 2 | `siav2-final-1280` / `siav2-p4p6-pruned-w250-1280` | 1280 | `cfg/deploy/yolov7-l6-siav2.yaml`; `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w250.yaml` | W6 improved: P3 removed, P4/P5/P6 only, width 0.25, anchor head kept | 2.1856 / 1.9552 | 2.15x | Yes | Final pre-training candidate |
| 1280 / W6 base | 3 | `siav2-dual-anchorfree-1280` | 1280 | `cfg/deploy/yolov7-l6-siav2-dual_anchorfree.yaml` | SIAV2 backbone/neck plus anchor-based branch and anchor-free branch together | 2.3912 / 2.0265 | 1.97x | No | Near 2x, latency-only |
| 1280 / W6 base | 4 | `siav2-p4p6-pruned-w300-1280` | 1280 | `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w300.yaml` | P3 removed, P4/P5/P6 only, width 0.30 | 2.4012 / 2.2189 | 1.96x | Candidate cfg only | Near miss |
| 1280 / W6 base | 5 | `siav2-dualhead-1280` | 1280 | `cfg/deploy/yolov7-l6-siav2-dualhead.yaml` | SIAV2 backbone/neck plus `DualDetect`, two anchor-based detection branches | 2.5697 / 2.3350 | 1.83x | No | Slower head ablation |
| 1280 / W6 base | 6 | `siav2-p4p6-pruned-w400-1280` | 1280 | `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w400.yaml` | P3 removed, P4/P5/P6 only, width 0.40 | 2.5868 / 2.3970 | 1.82x | Candidate cfg only | Below 2x |
| 1280 / W6 base | 7 | `siav2-p4p6-pruned-w600-1280` | 1280 | `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w600.yaml` | P3 removed, P4/P5/P6 only, width 0.60 | 3.1009 / 2.9509 | 1.52x | Candidate cfg only | Below 2x |
| 1280 / W6 base | 8 | `siav2-p4p6-pruned-w550-1280` | 1280 | `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w550.yaml` | P3 removed, P4/P5/P6 only, width 0.55 | 3.1152 / 2.9355 | 1.51x | Candidate cfg only | Below 2x |
| 1280 / W6 base | 9 | `siav2-4scale-w500-1280` | 1280 | `cfg/deploy/yolov7-l6-siav2-w500.yaml` | W6 4-scale kept, P3/P4/P5/P6 kept, width 0.50 | 3.1367 / 2.9608 | 1.50x | Candidate cfg only | P3 kept, not enough |
| 1280 / W6 base | 10 | `baseline-w6-1280` | 1280 | `cfg/deploy/yolov7-w6-nc16.yaml` | Baseline W6, `nc=16`, P3/P4/P5/P6 Detect | 4.7009 / 4.4584 | 1.00x | Yes | 1280 base |

## Generated But Not In Final TRT Latency Table

These cfgs exist, but they are not part of the final latency CSV used above.

| cfg | family | intended change | latency status |
|---|---|---|---|
| `cfg/deploy/yolov7-l6-siav2-w450.yaml` | 4-scale width sweep | W6-like 4-scale, width 0.45 | Not in final TRT table |
| `cfg/deploy/yolov7-l6-siav2-w550.yaml` | 4-scale width sweep | W6-like 4-scale, width 0.55 | Not in final TRT table |
| `cfg/deploy/yolov7-l6-siav2-w600.yaml` | 4-scale width sweep | W6-like 4-scale, width 0.60 | Not in final TRT table |
| `cfg/deploy/yolov7-l6-siav2-w1000.yaml` | 4-scale width sweep | W6-like 4-scale, width 1.00 | Not in final TRT table |
| `cfg/deploy/yolov7-l6-siav2-p4p6-lite-w250.yaml` | P4/P5/P6 lite sweep | P3 removed, lite routing, width 0.25 | Not in final TRT table |
| `cfg/deploy/yolov7-l6-siav2-p4p6-lite-w300.yaml` | P4/P5/P6 lite sweep | P3 removed, lite routing, width 0.30 | Not in final TRT table |
| `cfg/deploy/yolov7-l6-siav2-p4p6-lite-w400.yaml` | P4/P5/P6 lite sweep | P3 removed, lite routing, width 0.40 | Not in final TRT table |
| `cfg/deploy/yolov7-l6-siav2-p4p6-lite-w500.yaml` | P4/P5/P6 lite sweep | P3 removed, lite routing, width 0.50 | Not in final TRT table |
| `cfg/deploy/yolov7-l6-siav2-p4p6-lite-w550.yaml` | P4/P5/P6 lite sweep | P3 removed, lite routing, width 0.55 | Not in final TRT table |
| `cfg/deploy/yolov7-l6-siav2-p4p6-lite-w600.yaml` | P4/P5/P6 lite sweep | P3 removed, lite routing, width 0.60 | Not in final TRT table |
| `cfg/deploy/yolov7-l6-siav2-p4p6-lite-w650.yaml` | P4/P5/P6 lite sweep | P3 removed, lite routing, width 0.65 | Not in final TRT table |
| `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w500.yaml` | P4/P5/P6 pruned sweep | P3 removed, P4/P5/P6 only, width 0.50 | Not in final TRT table |
| `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w650.yaml` | P4/P5/P6 pruned sweep | P3 removed, P4/P5/P6 only, width 0.65 | Not in final TRT table |

## Weight Reference

| model | weight | size |
|---|---|---:|
| W6 deploy baseline | `weights/random-yolov7-w6-nc16-deploy.pt` | 267.29 MB |
| SIAV2 final deploy | `weights/random-yolov7-l6-siav2-deploy.pt` | 16.57 MB |
| SIAV2 training cfg random init | `weights/random-yolov7-l6-siav2.pt` | 19.31 MB |
| SIAV2 final P4/P5/P6 pruned w250 | `weights/random-yolov7-l6-siav2-p4p6-pruned-w250.pt` | 19.32 MB |
| SIAV2 dual head | `weights/random-yolov7-l6-siav2-dualhead.pt` | 16.71 MB |
| SIAV2 anchor-free | `weights/random-yolov7-l6-siav2-anchorfree.pt` | 16.48 MB |
| SIAV2 dual + anchor-free | `weights/random-yolov7-l6-siav2-dual_anchorfree.pt` | 16.62 MB |

## Final Selection

- Final train-ready candidate: `siav2-final-1280` / `siav2-p4p6-pruned-w250-1280`.
- Final candidate change: W6 improved by removing P3/8, keeping P4/P5/P6 only, reducing width to 0.25, keeping anchor-based Detect.
- W6 1280 baseline: 4.7009 / 4.4584 ms.
- Final SIAV2 1280: 2.1856 / 1.9552 ms.
- Final SIAV2 speedup: 2.15x avg, 2.28x median.
- Fastest latency-only model: `siav2-anchorfree-1280`, 2.0787 / 1.8218 ms, but it is not train-ready yet because loss/assignment is not implemented.

