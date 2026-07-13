# SIAV2 Distillation And Validation Plan

Date: 2026-07-13

## Scope

This document defines the next train-time accuracy-defense workflow. No training is run by this document.

Current status:

- `ComputeLossOTA` NameError is fixed.
- `P4/P5/P6 w250` and `P3-lite/P4/P5 w250` are both available as training candidates.
- W6 -> SIAV2 response distillation is implemented as an optional `train_aux.py` path.
- Optional cross-stride response distillation is implemented for the P3-removed candidate.
- Dataset EDA tooling is added to quantify small-object risk before training.

## 1. Distillation Design

Implemented first-stage distillation:

- Teacher: W6 nc16 checkpoint.
- Student: SIAV2 aux model.
- Matching rule: only detection heads with the same stride are distilled.
- Optional cross-stride rule: W6 stride 8 can supervise SIAV2 stride 16 by pooled response distillation.
- Loss type: raw response distillation.
- Default response terms:
  - objectness BCE on all matched cells.
  - class BCE on teacher-confident cells.
  - box-logit distillation is available but default-off.
- Optional small-object gain: multiply distillation loss when a batch contains boxes below `--distill-small-px`.

Why stride matching:

- `P4/P5/P6 w250` matches W6 on strides `16, 32, 64`.
- `P3-lite/P4/P5 w250` matches W6 on strides `8, 16, 32`.
- The same distillation code can train both candidates without hard-coded layer indices.
- `P4/P5/P6 w250` can additionally use `8:16` cross-stride distillation so removed P3 response is not completely ignored.

Implemented options in `train_aux.py`:

```text
--distill
--teacher-weights PATH
--distill-weight 0.25
--distill-obj-weight 1.0
--distill-cls-weight 1.0
--distill-box-weight 0.0
--distill-temp 2.0
--distill-conf-thres 0.01
--distill-strides 8 16 32
--distill-cross-strides 8:16
--distill-cross-weight 0.5
--distill-small-gain 1.25
--distill-small-px 128
```

Implementation files:

- `utils/distill.py`
- `train_aux.py`

Smoke verification completed:

```text
P3-lite student vs W6 teacher: student strides [8, 16, 32], teacher strides [8, 16, 32, 64]
P4/P5/P6 student vs W6 teacher: student strides [16, 32, 64], teacher strides [8, 16, 32, 64]
```

Both candidates produced a finite response distillation loss on synthetic input.

## 2. Candidate Training Matrix

| Candidate | CFG | Distill Strides | Purpose |
|---|---|---:|---|
| W6 teacher | `cfg/training/yolov7-w6-nc16.yaml` | none | baseline and teacher |
| SIAV2 P4/P5/P6 | `cfg/training/yolov7-l6-siav2-p4p6-pruned-w250.yaml` | `16 32 64` + cross `8:16` | speed-first model with P3 response compensation |
| SIAV2 P3-lite/P4/P5 | `cfg/training/yolov7-l6-siav2-p3lite-p4p5-w250.yaml` | `8 16 32` | small-object risk defense |

Optional speed-only reference:

- `cfg/deploy/yolov7-l6-siav2-p4p5-pruned-w250.yaml` has deploy latency, but no training cfg is selected by default because it does not restore P3.

## 3. Dataset EDA

Before training, run:

```powershell
conda run --no-capture-output -n yolov7 python tools\siav2_dataset_eda.py `
  --data data\siav2.yaml `
  --imgsz 1280 `
  --cfg cfg\training\yolov7-w6-nc16.yaml `
  --anchor-t 4.0 `
  --json-out runs\siav2\eda\siav2_dataset_eda.json `
  --md-out runs\siav2\eda\siav2_dataset_eda.md
```

Must inspect:

- total train/val image count.
- missing or empty labels.
- class imbalance.
- boxes per image.
- percentage of boxes with max side <= `32`, `64`, `96`, `128` pixels at 1280.
- anchor best possible recall.
- invalid label lines; this must be zero before training.

EDA smoke verification completed on `data/coco128.yaml`:

- train images: `128`
- train boxes: `929`
- max-side <= 64 px at 1280: `17.653%`
- max-side <= 128 px at 1280: `37.675%`
- W6 anchor BPR at `anchor_t=4.0`: `99.569%`

Interpretation:

- If many boxes are <= `64 px`, P3 removal risk is high.
- If anchor BPR is weak on real data, recalculate anchors before serious training.
- If class imbalance is severe, class-aware sampling or loss balancing may be needed.

## 4. Training Commands

Use:

```powershell
scripts\run_siav2_distill_candidates.ps1 `
  -Data data\siav2.yaml `
  -TeacherWeights runs\siav2_train\w6_nc16_teacher\weights\best.pt `
  -Device 0 `
  -Epochs 100 `
  -BatchSize 4 `
  -ImgSize 1280 `
  -Candidates p4p6,p3lite
```

The script runs:

- `P4/P5/P6 w250` with strides `16 32 64`.
- `P4/P5/P6 w250` additionally uses cross-stride `8:16` by default.
- `P3-lite/P4/P5 w250` with strides `8 16 32`.

Use `-Candidates p3lite` or `-Candidates p4p6` to run one candidate.

## 5. Validation Protocol

The production claim must be measured against a same-data W6 nc16 baseline:

```text
Delta mAP = SIAV2 mAP - W6 nc16 mAP
```

Required table:

| Model | mAP50 | mAP50-95 | small AP | medium AP | large AP | TRT FP16 1280 avg | TRT FP16 1280 median |
|---|---:|---:|---:|---:|---:|---:|---:|
| W6 nc16 teacher | | | | | | | |
| SIAV2 P4/P5/P6 distill | | | | | | | |
| SIAV2 P3-lite/P4/P5 distill | | | | | | | |

Use `tools/eval_siav2_size_ap.py` or `scripts/run_siav2_eval_matrix.ps1` to produce the small/medium/large AP rows.

Decision:

- If `P4/P5/P6` is within `3%p` and small AP is acceptable, select the speed-first model.
- If `P4/P5/P6` loses small AP but `P3-lite/P4/P5` stays within target, select P3-lite.
- If both miss by more than `3%p`, add stronger distillation or revisit INT8/width tradeoff.

## 6. Not Implemented Yet

Feature distillation is intentionally not added in this pass.

Reason:

- It requires stable teacher/student layer-pair selection and an adapter strategy for channel mismatch.
- Response distillation plus optional cross-stride response pooling is lower-risk and works across both selected candidates.

Recommended next step only if response distillation is insufficient:

- Add optional feature-stat distillation for P3/P4/P5 neck features.
- Use explicit layer pairs, e.g. student `77/78/89` vs teacher `83/93/103` for the P3-lite candidate.
