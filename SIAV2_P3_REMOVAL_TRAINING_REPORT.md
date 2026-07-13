# SIAV2 P3 Removal Training Report

Date: 2026-07-12

## Goal

Keep the speed advantage from removing P3 while reducing the expected small-object recall loss. PGI was not used. Training is based on `train_aux.py` and YOLOv7 aux-head training.

The practical target is:

1. Keep the P3-removed model near the 2x TensorRT FP16 speed target versus W6 1280.
2. Add P4-side compensation that does not add inference cost.
3. Accept equal or slightly lower accuracy than W6 only after a real full-data training run. On COCO128, treat the result as a training-path probe, not production accuracy.

## Implemented P3-Removal Countermeasures

| item | implementation | purpose |
|---|---|---|
| P4 small anchors | `cfg/training/yolov7-l6-siav2-p4small.yaml` and deploy equivalent | P3 is removed, so the first P4 head receives small-object anchors `[19,27, 44,40, 38,94]`. |
| COCO128 comparison cfg | `cfg/training/yolov7-l6-siav2-p4small-coco80.yaml` | Same structure as SIAV2 p4small, but `nc=80` for COCO128. |
| Small-target fallback assignment | `utils/loss.py`, `ComputeLossAuxOTA.add_small_target_fallback()` | If a small GT has no standard anchor match, force top-k closest anchors on P4 to become positives. |
| P4/P5/P6 object balance | `obj_balance` hyp key | Boost P4 loss weight because P4 now covers the smallest objects. |
| Close mosaic | `train_aux.py --close-mosaic` and hyp `close_mosaic` | Disable mosaic/mixup/copy-paste near the end of training. |
| Gradient clipping | `train_aux.py --grad-clip` | Stabilize long aux training. |
| TensorBoard off switch | `train_aux.py --no-tensorboard` | Avoid package noise and reduce logging overhead. |
| YOLOv7 checkpoint format | `tools/make_random_weights.py` | Random checkpoints now include `optimizer`, `epoch`, `best_fitness`, etc. |

## Candidate Hyp Files

| candidate | hyp | small target policy | P4 balance |
|---|---|---|---|
| base | `data/hyp.siav2-p4small-aux.yaml` | max 96 px, top-k 2 | `[6.0, 1.5, 0.5]` |
| conservative | `data/hyp.siav2-p4small-aux-conservative.yaml` | max 64 px, top-k 1 | `[5.0, 1.25, 0.5]` |
| relaxed | `data/hyp.siav2-p4small-aux-relaxed.yaml` | max 128 px, top-k 2, anchor_t 6 | `[7.0, 1.5, 0.5]` |

## Build And Smoke Tests

| check | result |
|---|---|
| `cfg/training/yolov7-l6-siav2-p4small.yaml` build | passed |
| `cfg/training/yolov7-l6-siav2-p4small-coco80.yaml` build | passed |
| `cfg/training/yolov7-w6.yaml` build | passed |
| dummy `ComputeLossAuxOTA` forward with small targets | passed |
| SIAV2 1280 1epoch smoke, batch 2 | passed, about 1.9 GB GPU memory |
| W6 1280 1epoch smoke, batch 1 | passed, but about 0.121 h for 1 epoch under current GPU load |

## COCO128 100epoch SIAV2 Results

Conditions:

- Dataset: `data/coco128.yaml`
- Image size: 1280
- Train script: `train_aux.py`
- Initial weights: `weights/random-yolov7-l6-siav2-p4small-coco80.pt`
- Batch: 2
- `--notest` used during training; final epoch validation still ran.

| run | final P | final R | final mAP@.5 | final mAP@.5:.95 | final train loss | final GPU mem | selected? |
|---|---:|---:|---:|---:|---:|---:|---|
| `siav2_p4small_aux_1280_e100` | 0.5704 | 0.003352 | 0.001330 | 0.000395 | 0.2680 | 2.35 GB | no |
| `siav2_p4small_aux_conservative_1280_e100` | 0.2396 | 0.000277 | 0.000025 | 0.000004 | 0.2520 | 2.12 GB | no |
| `siav2_p4small_aux_relaxed_1280_e100` | 0.2880 | 0.008362 | 0.002400 | 0.000487 | 0.2752 | 3.87 GB | yes |
| `w6_aux_1280_e100` | 0.0000295 | 0.002773 | 0.00000767 | 0.00000125 | 0.2046 | 7.01 GB | W6 reference |

Best SIAV2 candidate so far:

- Run: `runs/siav2_coco128_train/siav2_p4small_aux_relaxed_1280_e100`
- Best weight copied to: `weights/siav2-p4small-aux-relaxed-coco128-best.pt`
- Last weight copied to: `weights/siav2-p4small-aux-relaxed-coco128-last.pt`
- W6 reference best copied to: `weights/w6-coco128-e100-best.pt`
- W6 reference last copied to: `weights/w6-coco128-e100-last.pt`

## TensorRT FP16 Latency Reference

Latency uses the nc16 deploy graphs and random weights, batch 1. The COCO128 training runs above use nc80 only for the public COCO128 probe.

| model | resolution | graph | avg / median latency | speed vs W6 1280 | status |
|---|---:|---|---:|---:|---|
| `baseline-w6-1280` | 1280 | P3/P4/P5/P6 W6 | 4.7009 / 4.4584 ms | 1.00x | baseline |
| `siav2-p4p6-pruned-w250-1280` | 1280 | P3 removed, P4/P5/P6 only, width 0.25, anchor Detect | 2.1856 / 1.9552 ms | 2.15x avg, 2.28x median | train-ready family |
| `siav2-anchorfree-1280` | 1280 | anchor-free proxy head | 2.0787 / 1.8218 ms | 2.26x avg | latency-only, no train loss yet |

Layer-profile summary from `runs/siav2/trt/profile_summary_final.csv`:

| profile | backbone ms | neck ms | detect/decode ms | other ms |
|---|---:|---:|---:|---:|
| `yolov7-w6-nc16` | 2.0436 | 1.5752 | 0.5871 | 0.4949 |
| `siav2-p4p6-pruned-w250` | 0.8214 | 0.9557 | 0.0000 | 0.4086 |

## Interpretation

The relaxed fallback is the best of the three P3-removal countermeasures on COCO128. It also beats the W6 random-init 100epoch COCO128 reference on final mAP in this probe:

| comparison | mAP@.5 | mAP@.5:.95 | read |
|---|---:|---:|---|
| SIAV2 relaxed | 0.002400 | 0.000487 | best candidate in this probe |
| W6 reference | 0.00000767 | 0.00000125 | lower final mAP despite lower train loss |

The absolute mAP is still very low, so this is not a production quality success. It proves that:

- The P3-removed aux training path runs.
- Small-target fallback does not break loss/assignment.
- The relaxed policy is less bad than base/conservative on this 100epoch COCO128 probe.
- Under this tiny random-init probe, the smaller SIAV2 model overfits/learns faster than W6.
- A real decision on "equal or slightly lower than W6" needs full dataset training from a meaningful initialization.

## P3 Removal Decision

P3 removal is still the correct speed-side move. Re-adding a real stride-8/P3 head would directly attack the 2x latency target. The replacement strategy should stay inference-free or near-free:

| countermeasure | inference cost | decision |
|---|---:|---|
| P4 small anchors | none/negligible | keep |
| small-target fallback assignment in aux loss | none, train-only | keep |
| P4 objectness balance | none, train-only | keep |
| relaxed `anchor_t` for small objects | none, train-only | keep as current best |
| dual head | measurable latency cost | not selected for final train-ready path |
| anchor-free head | fastest latency-only path | revisit only after implementing real loss/assignment |

Final pre-full-training candidate:

- Architecture/deploy family: `siav2-p4p6-pruned-w250-1280`
- Training cfg: `cfg/training/yolov7-l6-siav2-p4small-coco80.yaml` for COCO128 probe; `cfg/training/yolov7-l6-siav2-p4small.yaml` for SIAV2 nc16.
- Hyp: `data/hyp.siav2-p4small-aux-relaxed.yaml`
- COCO128 selected weight: `weights/siav2-p4small-aux-relaxed-coco128-best.pt`

## W6 Comparison Status

Completed.

- Run: `runs/siav2_coco128_train/w6_aux_1280_e100`
- Log: `runs/siav2_coco128_train/logs/w6_aux_1280_e100.log`
- Runtime: 100 epochs completed in about 1.002 hours.
- Final weight: `weights/w6-coco128-e100-best.pt`
