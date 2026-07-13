# SIAV2 작업지시서 - 학습 직전까지 완료

대상: 다른 PC의 AI 에이전트  
기준 경로: YOLOv7 저장소 루트

## 1. 최우선 지시

먼저 `SIAV2_DESIGN.md`를 처음부터 끝까지 읽어라. 그 문서가 설계, 목표, 판단 기준의 원본이다. 이 문서는 이번 작업 범위와 중단 지점을 고정하는 실행 지시서다.

`SIAV2_DESIGN.md`가 없거나 일부만 있으면 즉시 멈추고 보고하라. 추측으로 설계 의도를 보완하지 마라.

## 2. 절대 경계

이번 임무는 **학습 직전까지**다. 아래 명령과 동작은 금지다.

- `python train.py ...`
- `python train_aux.py ...`
- 학습 루프, optimizer step, backward pass, epoch 실행
- 실제 weight를 개선하는 fine-tuning, pretraining, resume training
- INT8 calibration 또는 TensorRT engine 생성처럼 별도 산출물을 고정하는 최적화 작업

랜덤 weight 생성, cfg build, dry-run forward, latency 측정, 데이터셋 EDA, anchor 재계산 준비, 평가 프로토콜 고정은 허용된다.

## 3. GPU 사용량 제한

이 저장소 작업으로 발생하는 GPU 사용량은 **항상 30% 미만**으로 유지하라. `nvidia-smi` 기준 `GPU-Util`이 30% 이상으로 올라가면 해당 명령을 중단하고 조건을 낮추거나 CPU 경로로 전환하라.

권장 운용 원칙:

- 가능한 모든 cfg build와 구조 검증은 `--device cpu`로 먼저 수행한다.
- GPU가 꼭 필요한 latency 검증은 단일 GPU, 단일 프로세스, `batch-size=1`, 짧은 warmup/repeat로만 수행한다.
- 동시에 다른 학습, 추론, TensorRT 변환, 데이터로더 stress job을 실행하지 않는다.
- PowerShell에서는 작업 전 `$env:CUDA_VISIBLE_DEVICES="0"`처럼 대상 GPU를 하나로 제한한다.
- 모니터링 로그를 남긴다.

예시 모니터링:

```powershell
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv -l 1
```

latency 측정 자체가 30% 미만 조건과 양립하지 않으면 30% 제한을 우선한다. 그 경우 Phase 0 속도 게이트는 미판정으로 두고, `SIAV2_PHASE0_1_REPORT.md`에 "GPU cap conflict"로 보고하라. 제한을 넘겨서 통과 판정을 만들지 마라.

## 4. Definition of Done

아래 항목을 모두 만족하면 학습 직전 준비 완료로 본다.

- `SIAV2_DESIGN.md`를 정독했고 핵심 파라미터를 보고서에 재기록했다.
- torch, CUDA, GPU, Python package 환경 sanity check를 완료했다.
- `cfg/training/yolov7-l6-siav2.yaml`가 존재하고 `nc=16` 기준으로 build된다.
- W6 baseline cfg와 SIAv2 cfg가 모두 build 또는 dry-run forward를 통과한다.
- 랜덤 weight 기반 Phase 0 latency 비교를 시도했고, 2배 속도 게이트를 판정하거나 GPU 30% 제한 때문에 미판정으로 보고했다.
- Phase 1 준비 항목인 데이터 EDA, label sanity, head 변형 cfg build 검증, anchor 준비, 평가 프로토콜 동결을 완료했다.
- 어떤 학습 스크립트도 실행하지 않았다.
- `SIAV2_PHASE0_1_REPORT.md`를 작성했다.

확정 파라미터:

- 입력 크기: `1280`
- 클래스 수: `nc=16`
- Phase 0 속도 목표: W6 baseline 대비 SIAv2 latency `<= 0.50x`
- 기준 GPU: RTX A4000 또는 설계 문서가 지정한 동급 GPU
- GPU 사용량 제한: 이 저장소 작업 기준 `GPU-Util < 30%`

## 5. 포함 범위

이번에 수행할 작업:

- 환경 sanity check
- 필수 파일 존재 확인
- cfg build 검증
- 랜덤 weight 생성 또는 기존 랜덤 weight 검증
- Phase 0 latency gate 시도
- Phase 1 학습 준비 검증
- 보고서 작성

이번에 수행하지 않을 작업:

- 모든 형태의 train 실행
- mAP baseline을 만들기 위한 학습
- anchor-free 전환
- INT8, TensorRT engine 생성, production export
- 하이퍼파라미터 탐색 학습

## 6. Task 0 - 환경 및 파일 sanity

저장소 루트에서 실행한다.

```powershell
python --version
python -m pip --version
nvidia-smi
```

torch/CUDA 확인:

```powershell
@'
import sys
import torch
print("python", sys.version)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
'@ | python -
```

필수 파일 확인:

```powershell
Test-Path SIAV2_DESIGN.md
Test-Path cfg/training/yolov7-l6-siav2.yaml
Test-Path tools/make_random_weights.py
Test-Path data/siav2.yaml
```

완료 조건:

- torch import가 성공한다.
- CUDA 사용 가능 여부와 GPU 이름을 기록한다.
- 필수 파일 누락 여부를 명확히 기록한다.
- 누락 파일이 있으면 `SIAV2_DESIGN.md` 지시에 따라 생성 가능한지 판단한다. 설계 문서 없이 재구성하지 마라.

## 7. Task 1 - cfg build 검증

CPU build를 먼저 수행한다.

```powershell
python models/yolo.py --cfg cfg/training/yolov7-w6.yaml --device cpu
python models/yolo.py --cfg cfg/training/yolov7-l6-siav2.yaml --device cpu
```

CPU build가 성공하고 GPU dry-run이 필요하면 30% GPU 제한을 지키면서 최소 입력으로만 확인한다. GPU 사용량이 30% 이상이면 즉시 중단하고 CPU build 결과까지만 보고한다.

검증 포인트:

- parser error 없음
- layer index routing error 없음
- Detect head shape mismatch 없음
- `nc=16` 반영 확인
- stride, anchors, output tensor shape를 보고서에 기록

## 8. Task 2 - 랜덤 weight 준비

`tools/make_random_weights.py --help`를 먼저 확인하고, 실제 스크립트의 인자에 맞춰 실행한다. 예시는 아래와 같다.

```powershell
python tools/make_random_weights.py --cfg cfg/training/yolov7-w6.yaml --out weights/random-yolov7-w6-nc16.pt --nc 16 --device cpu
python tools/make_random_weights.py --cfg cfg/training/yolov7-l6-siav2.yaml --out weights/random-yolov7-l6-siav2-nc16.pt --nc 16 --device cpu
```

완료 조건:

- W6 baseline과 SIAv2 랜덤 weight가 모두 생성되거나, 이미 존재한다면 metadata를 검증했다.
- weight 생성은 학습이 아니어야 한다.
- 생성 파일 경로, cfg, seed, nc 값을 보고서에 기록한다.

## 9. Task 3 - Phase 0 latency gate

목표는 학습 없이 구조 변경만으로 W6 대비 SIAv2가 2배 빠를 가능성이 있는지 확인하는 것이다.

원칙:

- 입력 크기 `1280`
- batch size `1`
- 랜덤 weight만 사용
- 동일 장비, 동일 device, 동일 precision 조건
- GPU 사용량 30% 미만 유지
- 측정 중 30% 이상이면 중단하고 미판정 처리

가능하면 설계 문서에 지정된 benchmark 스크립트를 우선 사용한다. 지정이 없으면 `test.py --task speed` 또는 별도 latency dry-run 스크립트를 사용하되, 스크립트 작성 시 forward-only, no-grad, eval mode, fixed warmup/repeat만 포함한다.

판정:

- PASS: SIAv2 median latency `<= W6 median latency * 0.50`
- FAIL / Trigger A: SIAv2 median latency `> W6 median latency * 0.50`
- UNDECIDED: GPU 30% 제한, 환경 문제, 필수 파일 누락 때문에 공정 측정 불가

Trigger A가 발생하면 Phase 1 학습 준비는 완료하되, 학습으로 넘어가지 말고 보고서에서 "프로젝트 존폐 게이트 미달"로 표시한다.

## 10. Task 4 - Phase 1 학습 준비

학습 실행 없이 준비만 완료한다.

데이터 EDA:

- `data/siav2.yaml` 경로 확인
- train/val 이미지 수
- class 수와 class 이름 수가 `nc=16`과 일치하는지 확인
- label 파일 누락, 빈 label, 잘못된 class id, box 좌표 범위 오류 확인
- class별 instance count와 imbalance 기록

head 변형 cfg 검증:

- `SIAV2_DESIGN.md`가 지정한 head 변형 cfg를 생성 또는 확인한다.
- 각 변형은 `python models/yolo.py --cfg ... --device cpu` build를 통과해야 한다.
- routing 재구성이 필요한 경우 변경 이유와 layer index를 기록한다.
- 학습으로 성능을 확인하지 않는다.

anchor 준비:

- 설계 문서가 지정한 anchor 재계산 절차를 준비한다.
- label 통계와 anchor 후보를 기록한다.
- anchor 성능 검증을 위해 train을 실행하지 않는다.

평가 프로토콜 동결:

- validation split
- img size `1280`
- conf/iou threshold
- batch size
- precision 조건
- 저장 경로와 naming convention

## 11. STOP 지점

아래까지 끝나면 즉시 멈춘다.

- 환경 sanity 완료
- cfg build 완료
- Phase 0 latency gate 판정 또는 미판정 사유 기록
- Phase 1 학습 준비 완료
- 보고서 작성 완료

그 다음 작업은 사람 승인 후 별도 지시로만 진행한다. 절대 `train.py` 또는 `train_aux.py`를 실행하지 마라.

## 12. 보고서 작성

`SIAV2_PHASE0_1_REPORT.md`를 작성한다. 최소 포함 항목:

- 실행 PC 사양: OS, Python, torch, CUDA, GPU, driver
- GPU 30% 제한 준수 방법과 `nvidia-smi` 관찰 결과
- 필수 파일 존재 여부
- `SIAV2_DESIGN.md`에서 확인한 핵심 설계 요약
- cfg build 결과 표
- 랜덤 weight 생성/검증 결과
- Phase 0 latency 표: W6, SIAv2, ratio, PASS/FAIL/UNDECIDED
- Trigger A 여부
- 데이터 EDA 요약
- label 오류와 수정 여부
- head 변형 cfg build 결과
- anchor 준비 결과
- 평가 프로토콜
- 실행한 명령 목록
- 실행하지 않은 금지 명령 확인: `train.py`, `train_aux.py`
- 다음 단계 승인 요청

## 13. 문제 발생 시 처리

| 상황 | 처리 |
| --- | --- |
| `SIAV2_DESIGN.md` 없음 | 즉시 STOP. 설계 문서 요청. |
| SIAv2 cfg 없음 | 설계 문서에 근거해 생성 가능하면 생성, 아니면 STOP. |
| cfg build 실패 | stack trace, layer index, shape를 기록하고 가능한 최소 수정 후 재검증. |
| GPU 사용량 30% 이상 | 명령 중단. batch/repeat 축소 또는 CPU 검증으로 전환. latency는 UNDECIDED 처리. |
| 2배 속도 미달 | Trigger A로 보고. 학습 금지. |
| data yaml 없음 | EDA 불가로 보고. 학습 금지. |
| label 오류 다수 | 오류 샘플과 count 보고. 자동 수정은 설계 문서 또는 명시 지시가 있을 때만 수행. |
| torch/CUDA 불량 | 환경 문제로 보고. cfg CPU build가 가능하면 그것까지만 수행. |

## 14. 최종 한 줄 상태

보고서 마지막에 아래 중 하나를 반드시 적는다.

- `READY_FOR_TRAINING_APPROVAL`: 학습 직전 준비 완료, 사람 승인 대기
- `BLOCKED_MISSING_INPUT`: 필수 문서/파일/데이터 누락
- `BLOCKED_ENVIRONMENT`: torch/CUDA/GPU 환경 문제
- `TRIGGER_A_SPEED_GATE_FAILED`: 2배 속도 게이트 실패
- `UNDECIDED_GPU_CAP_CONFLICT`: GPU 30% 제한 때문에 latency 공정 측정 불가
