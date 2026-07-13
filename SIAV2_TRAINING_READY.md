# SIAV2 Training-Ready Workflow

Date: 2026-07-13

This repo now contains the pieces needed to train and compare the SIAV2 candidates once `data/siav2.yaml` and the dataset are present.

## What Is Ready

- W6 nc16 teacher/baseline training entrypoint.
- SIAV2 P4/P5/P6 w250 distillation candidate.
- SIAV2 P3-lite/P4/P5 w250 distillation candidate.
- P4/P5/P6 P3 compensation via optional cross-stride response distillation `8:16`.
- Dataset EDA with label validity checks.
- Size-bucket AP evaluation for all/small/medium/large targets.
- Full PowerShell workflow wrapper.

## Required Inputs

- `data/siav2.yaml`
- SIAV2 YOLO labels with class ids `0..15`
- Optional W6 initialization weights, if available

## 1. Dataset Gate

```powershell
conda run --no-capture-output -n yolov7 python tools\siav2_dataset_eda.py `
  --data data\siav2.yaml `
  --imgsz 1280 `
  --cfg cfg\training\yolov7-w6-nc16.yaml `
  --anchor-t 4.0 `
  --json-out runs\siav2\eda\siav2_dataset_eda.json `
  --md-out runs\siav2\eda\siav2_dataset_eda.md `
  --fail-on-invalid
```

Training should not start if `invalid_label_lines` is nonzero. Missing label files are reported separately because they may represent valid negative images.

## 2. Train W6 Teacher

```powershell
scripts\run_siav2_w6_teacher.ps1 `
  -Data data\siav2.yaml `
  -Device 0 `
  -Epochs 100 `
  -BatchSize 2 `
  -ImgSize 1280
```

Output teacher path:

```text
runs\siav2_train\w6_nc16_teacher\weights\best.pt
```

## 3. Train SIAV2 Candidates

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

Candidate behavior:

| Candidate | CFG | Hyp | Distillation |
|---|---|---|---|
| P4/P5/P6 w250 | `cfg/training/yolov7-l6-siav2-p4p6-pruned-w250.yaml` | `data/hyp.siav2-p4small-aux-relaxed.yaml` | same-stride `16 32 64` + cross `8:16` |
| P3-lite/P4/P5 w250 | `cfg/training/yolov7-l6-siav2-p3lite-p4p5-w250.yaml` | `data/hyp.siav2-p3lite-aux-relaxed.yaml` | same-stride `8 16 32` |

## 4. Evaluate Accuracy By Size

```powershell
scripts\run_siav2_eval_matrix.ps1 `
  -Data data\siav2.yaml `
  -Device 0 `
  -BatchSize 8 `
  -ImgSize 1280
```

This writes one `size_ap.json` and `size_ap.md` per model under `runs\siav2_eval`.

Decision table to fill:

| Model | mAP50 | mAP50-95 | small AP | medium AP | large AP | TRT FP16 1280 avg | TRT FP16 1280 median |
|---|---:|---:|---:|---:|---:|---:|---:|
| W6 nc16 teacher | | | | | | | |
| SIAV2 P4/P5/P6 distill | | | | | | | |
| SIAV2 P3-lite/P4/P5 distill | | | | | | | |

## 5. One-Command Workflow

```powershell
scripts\run_siav2_training_workflow.ps1 `
  -Data data\siav2.yaml `
  -Device 0 `
  -TeacherEpochs 100 `
  -StudentEpochs 100 `
  -TeacherBatchSize 2 `
  -StudentBatchSize 4 `
  -EvalBatchSize 8
```

Useful switches:

- `-SkipTeacher` if W6 teacher already exists.
- `-SkipStudents` to only run EDA/teacher/eval.
- `-SkipEval` to stop after training.
- `-NoAutoAnchor` only when anchors are intentionally frozen.

## Current Guardrails

- P4/P5/P6 remains the fastest candidate, but it still removes the real P3 head.
- Cross-stride `8:16` distillation is a training-time compensation, not a full replacement for stride-8 inference.
- If small AP drops too much, select P3-lite or revisit width/INT8 tradeoff.
- Feature distillation is still not implemented; add it only if response/cross-response distillation is insufficient.
