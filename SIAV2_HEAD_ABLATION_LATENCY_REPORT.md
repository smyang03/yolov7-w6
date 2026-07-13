# SIAV2 Head Ablation Latency Report

작성일: 2026-07-11  
조건: random weight, TensorRT FP16, `trtexec --fp16 --dumpProfile --noDataTransfers`, 학습 실행 없음

## 최종 베이스를 어떻게 만들었나

YOLOv7 large 계열 기준은 `yolov7-w6`로 잡았다. 공정 비교를 위해 baseline도 `nc=16`으로 맞춘 `cfg/deploy/yolov7-w6-nc16.yaml`을 만들었다.

최종 SIAV2 베이스:

- deploy cfg: `cfg/deploy/yolov7-l6-siav2.yaml`
- training cfg: `cfg/training/yolov7-l6-siav2.yaml`
- 구조: W6 파생, P3/8 제거, P4/P5/P6 3-scale head
- width: `0.25`
- nc: `16`
- stride: `[16, 32, 64]`
- deploy params: `4,269,661`
- training params: `4,985,575`

핵심 변경:

- W6 원본 4-scale `[8,16,32,64]`에서 P3/8 head를 제거했다.
- detection 후보 수를 줄였다.
- 채널 폭을 `0.25`까지 낮췄다.
- 3-scale aux head가 build되도록 `IAuxDetect` stride hardcode를 `m.nl` 기준으로 수정했다.

## 640 / 1280 출력 크기

| model | img | stride | output candidates | output shape |
| --- | ---: | --- | ---: | --- |
| W6-nc16 deploy | 640 | `[8,16,32,64]` | 25,500 | `[1,25500,21]` |
| SIAV2 final | 640 | `[16,32,64]` | 6,300 | `[1,6300,21]` |
| W6-nc16 deploy | 1280 | `[8,16,32,64]` | 102,000 | `[1,102000,21]` |
| SIAV2 final | 1280 | `[16,32,64]` | 25,200 | `[1,25200,21]` |

## Head 후보 정의

모든 후보는 같은 SIAV2 final backbone/neck을 공유하고 마지막 head만 바꿨다.

| 후보 | cfg | 의미 |
| --- | --- | --- |
| final | `cfg/deploy/yolov7-l6-siav2.yaml` | 3-scale anchor-based Detect |
| dual-head | `cfg/deploy/yolov7-l6-siav2-dualhead.yaml` | 같은 feature에서 anchor-based detect branch 2개 |
| anchor-free | `cfg/deploy/yolov7-l6-siav2-anchorfree.yaml` | scale당 anchor 3개 대신 grid당 1 prediction |
| dual+anchor-free | `cfg/deploy/yolov7-l6-siav2-dual_anchorfree.yaml` | anchor-based branch + anchor-free branch |

주의: anchor-free와 dual 계열은 latency/weight 검토용 head다. 학습 loss는 아직 연결하지 않았다.

## Random Weight

| model | weight | size MB |
| --- | --- | ---: |
| W6-nc16 deploy | `weights/random-yolov7-w6-nc16-deploy.pt` | 267.29 |
| SIAV2 final deploy | `weights/random-yolov7-l6-siav2-deploy.pt` | 16.57 |
| SIAV2 dual-head | `weights/random-yolov7-l6-siav2-dualhead.pt` | 16.71 |
| SIAV2 anchor-free | `weights/random-yolov7-l6-siav2-anchorfree.pt` | 16.48 |
| SIAV2 dual+anchor-free | `weights/random-yolov7-l6-siav2-dual_anchorfree.pt` | 16.62 |

## TRT FP16 Latency - 640

| model | avg ms | median ms | avg ratio vs W6 | median ratio vs W6 |
| --- | ---: | ---: | ---: | ---: |
| W6-nc16 | 3.2888 | 2.8132 | 1.0000 | 1.0000 |
| SIAV2 final | 2.0827 | 1.7859 | 0.6333 | 0.6348 |
| SIAV2 dual-head | 2.1300 | 1.8129 | 0.6477 | 0.6444 |
| SIAV2 anchor-free | 1.9239 | 1.7450 | 0.5850 | 0.6203 |
| SIAV2 dual+anchor-free | 1.9986 | 1.7764 | 0.6077 | 0.6315 |

640에서는 2배 gate에는 못 미친다. 가장 빠른 것은 anchor-free 단독이다.

## TRT FP16 Latency - 1280

| model | avg ms | median ms | avg ratio vs W6 | median ratio vs W6 |
| --- | ---: | ---: | ---: | ---: |
| W6-nc16 | 4.7009 | 4.4584 | 1.0000 | 1.0000 |
| SIAV2 final | 2.1856 | 1.9552 | 0.4649 | 0.4385 |
| SIAV2 dual-head | 2.5697 | 2.3350 | 0.5466 | 0.5237 |
| SIAV2 anchor-free | 2.0787 | 1.8218 | 0.4422 | 0.4086 |
| SIAV2 dual+anchor-free | 2.3912 | 2.0265 | 0.5087 | 0.4545 |

1280에서는 final과 anchor-free 단독이 2배 gate를 통과한다. dual-head는 head branch 증가 때문에 실패한다. dual+anchor-free는 median은 통과하지만 avg 기준으로는 살짝 실패한다.

## 판단

현재 latency만 보면:

- 640 최속: `anchor-free`
- 1280 최속: `anchor-free`
- 1280 안정 통과 후보: `anchor-free`, `final`
- dual-head는 latency 이득이 없고 weight도 약간 증가한다.
- dual+anchor-free는 final보다 느리고 avg 기준 2배 gate에서 경계선이다.

다음 단계에서 mAP/recall을 보려면 anchor-free loss와 target assignment를 구현해야 한다. 현재 산출물은 학습 전 latency/weight 리포트까지다.
