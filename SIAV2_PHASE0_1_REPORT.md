# SIAV2 Phase 0/1 Report - 학습 직전 검증

작성일: 2026-07-11  
환경: Windows, conda env `yolov7`, Python 3.8, torch `1.13.1+cu116`, TensorRT `10.7.0` Python / `trtexec` TensorRT `10.14.1`, GPU `NVIDIA GeForce RTX 4090`

Latency policy update: official SIAV2 latency evidence is accepted only from `trtexec` TensorRT `8.6.1.x` or `10.14.x`. The Python TensorRT binding version in this historical environment line is not used as a latency basis.

## 결론

TRT FP16 기준 2배 latency gate를 통과하는 후보를 찾았다.

최종 후보:

- training cfg: `cfg/training/yolov7-l6-siav2.yaml`
- deploy cfg: `cfg/deploy/yolov7-l6-siav2.yaml`
- 구조: YOLOv7-W6 파생, `nc=16`, `width_multiple=0.25`, P4/P5/P6 3-scale head, stride `[16, 32, 64]`
- training params: `4,985,575`
- deploy params: `4,269,661`
- output candidates at `1280`: `25,200` boxes, shape `[1, 25200, 21]`
- random checkpoint: `weights/random-yolov7-l6-siav2.pt`

주의: P3/8 head를 제거했기 때문에 작은 객체 recall 리스크가 있다. 속도 목표를 맞추기 위한 구조적 tradeoff다.

## TRT FP16 Latency Gate

측정 조건:

- input: `1x3x1280x1280`
- mode: TensorRT FP16
- command base: `trtexec --fp16 --dumpProfile --profilingVerbosity=detailed --noDataTransfers`
- baseline: `cfg/deploy/yolov7-w6-nc16.yaml`
- logs: `runs/siav2/trt/*.trtexec.log`
- profile JSON: `runs/siav2/trt/*.profile.json`

| model | avg ms | median ms | avg ratio | median ratio | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| yolov7-w6-nc16 | 4.7009 | 4.4584 | 1.0000 | 1.0000 | baseline |
| siav2-4scale-w500 | 3.1367 | 2.9608 | 0.6673 | 0.6641 | FAIL |
| siav2-p4p6-pruned-w600 | 3.1009 | 2.9509 | 0.6596 | 0.6619 | FAIL |
| siav2-p4p6-pruned-w550 | 3.1152 | 2.9355 | 0.6627 | 0.6584 | FAIL |
| siav2-p4p6-pruned-w400 | 2.5868 | 2.3970 | 0.5503 | 0.5376 | FAIL |
| siav2-p4p6-pruned-w300 | 2.4012 | 2.2189 | 0.5108 | 0.4977 | median-only PASS |
| siav2-p4p6-pruned-w250 | 2.1856 | 1.9552 | 0.4649 | 0.4385 | PASS |

PASS 기준: avg와 median 모두 W6-nc16 baseline 대비 `<= 0.50x`.

## Layer Profile 해석

초기 PyTorch forward 측정에서는 `w500`이 통과처럼 보였지만, TRT FP16에서는 실패했다. TRT layer profile상 병목은 detect/decode보다 backbone/neck conv에 더 강하게 남았다.

Baseline stage 합:

- backbone: 약 `2.04 ms`
- neck: 약 `1.58 ms`
- detect/decode: 약 `0.59 ms`
- other/copy/slice: 약 `0.49 ms`

최종 `w250` stage 합:

- backbone: 약 `0.82 ms`
- neck/detect 포함: 약 `0.96 ms`
- other/copy/slice: 약 `0.41 ms`

개선 판단:

- 단순 `nc=16` 또는 detect class 수 감소만으로는 부족했다.
- 4-scale `w500`은 TRT avg ratio `0.6673`으로 실패했다.
- P3 제거만으로도 부족했다.
- P4/P5/P6 3-scale + width `0.25`까지 낮춰야 TRT FP16 평균 기준 2배를 통과했다.

## Build / Forward 검증

검증 명령:

```powershell
conda run -n yolov7 python tools/inspect_cfgs.py --img 1280 --cfg cfg/training/yolov7-l6-siav2.yaml cfg/deploy/yolov7-l6-siav2.yaml cfg/training/yolov7-w6-nc16.yaml cfg/deploy/yolov7-w6-nc16.yaml
```

결과:

- `cfg/training/yolov7-l6-siav2.yaml`: build PASS, forward PASS, stride `[16,32,64]`
- `cfg/deploy/yolov7-l6-siav2.yaml`: build PASS, forward PASS, stride `[16,32,64]`
- `cfg/training/yolov7-w6-nc16.yaml`: build PASS, forward PASS, stride `[8,16,32,64]`
- `cfg/deploy/yolov7-w6-nc16.yaml`: build PASS, forward PASS, stride `[8,16,32,64]`

## 코드 수정

수정 파일:

- `models/yolo.py`
  - `Model.fuse()`를 `torch.no_grad()`로 감싸 torch 1.13 leaf in-place 오류를 해결했다.
  - `IAuxDetect` stride 계산의 `[:4]` 하드코딩을 `[:m.nl]`로 바꿔 3-scale aux head를 지원하게 했다.

추가 도구:

- `tools/make_siav2_variants.py`
- `tools/make_siav2_p4p6_variants.py`
- `tools/make_random_weights.py`
- `tools/inspect_cfgs.py`
- `tools/export_trt_onnx.py`
- `tools/benchmark_latency.py`
- `tools/analyze_trt_profile.py`

## 생성 산출물

최종:

- `cfg/training/yolov7-l6-siav2.yaml`
- `cfg/deploy/yolov7-l6-siav2.yaml`
- `weights/random-yolov7-l6-siav2.pt`
- `runs/siav2/onnx/siav2-p4p6-pruned-w250.onnx`
- `runs/siav2/trt/siav2-p4p6-pruned-w250.fp16.engine`
- `runs/siav2/trt/latency_summary.csv`

비교 후보:

- `cfg/deploy/yolov7-l6-siav2-w500.yaml`
- `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w600.yaml`
- `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w550.yaml`
- `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w400.yaml`
- `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w300.yaml`
- `cfg/deploy/yolov7-l6-siav2-p4p6-pruned-w250.yaml`

## 미완료 / 차단 항목

현재 저장소에 아래 파일이 없다.

- `SIAV2_DESIGN.md`
- `data/siav2.yaml`

따라서 다음 항목은 완료하지 못했다.

- 설계 문서 원문 대비 검증
- SIAV2 dataset EDA
- label sanity check
- class distribution 분석
- anchor 재계산 또는 anchor 적합도 검증

현재 anchor는 W6 P4/P5/P6 anchor를 그대로 사용한다.

## 학습 금지 확인

실행하지 않음:

- `python train.py ...`
- `python train_aux.py ...`
- optimizer step
- backward pass
- epoch 실행

수행한 작업은 build, no-grad forward, random checkpoint 생성, ONNX export, TensorRT FP16 profiling이다.

## 다음 단계

학습 승인 전에 필요한 입력:

- `SIAV2_DESIGN.md`
- `data/siav2.yaml`
- 실제 dataset 경로

입력이 들어오면 해야 할 일:

- class 수와 names가 `nc=16`인지 확인
- label 좌표/class id sanity check
- P3 제거가 작은 객체 recall에 미치는 리스크를 bbox 크기 분포로 확인
- P4/P5/P6 anchor 재계산 또는 기존 anchor 적합도 검증
- 문제가 없으면 사람 승인 후에만 학습 시작

최종 상태: `BLOCKED_MISSING_INPUT_WITH_TRT_SPEED_GATE_PASS`
