import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def tag(width):
    return f"w{int(round(width * 1000)):03d}"


def write_cfg(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False, width=120)


def make_lite(base, nc, width):
    cfg = deepcopy(base)
    cfg["nc"] = nc
    cfg["width_multiple"] = float(width)
    cfg["anchors"] = deepcopy(base["anchors"][1:])  # P4/P5/P6
    # Original deploy W6 tail is P3/P4/P5/P6 convs followed by Detect.
    # Drop only P3 detect conv. The PAN path is preserved.
    cfg["head"] = deepcopy(base["head"][:-5] + base["head"][-4:-1])
    cfg["head"].append([[114, 115, 116], 1, "Detect", ["nc", "anchors"]])
    return cfg


def make_pruned(base, nc, width):
    cfg = deepcopy(base)
    cfg["nc"] = nc
    cfg["width_multiple"] = float(width)
    cfg["anchors"] = deepcopy(base["anchors"][1:])  # P4/P5/P6
    # Keep the top-down path through global layer 71 (P4). Remove the P3 branch
    # and rebuild the bottom-up path directly from P4 to P5/P6.
    keep_head_count = 71 - len(base["backbone"]) + 1
    head = deepcopy(base["head"][:keep_head_count])
    head.extend(
        [
            [71, 1, "Conv", [512, 3, 1]],  # 72 P4 detect feature
            [71, 1, "Conv", [384, 3, 2]],  # 73 P4 -> P5
            [[-1, 59], 1, "Concat", [1]],  # 74
            [-1, 1, "Conv", [384, 1, 1]],  # 75
            [-2, 1, "Conv", [384, 1, 1]],  # 76
            [-1, 1, "Conv", [192, 3, 1]],  # 77
            [-1, 1, "Conv", [192, 3, 1]],  # 78
            [-1, 1, "Conv", [192, 3, 1]],  # 79
            [-1, 1, "Conv", [192, 3, 1]],  # 80
            [[-1, -2, -3, -4, -5, -6], 1, "Concat", [1]],  # 81
            [-1, 1, "Conv", [384, 1, 1]],  # 82 P5
            [-1, 1, "Conv", [512, 3, 2]],  # 83 P5 -> P6
            [[-1, 47], 1, "Concat", [1]],  # 84
            [-1, 1, "Conv", [512, 1, 1]],  # 85
            [-2, 1, "Conv", [512, 1, 1]],  # 86
            [-1, 1, "Conv", [256, 3, 1]],  # 87
            [-1, 1, "Conv", [256, 3, 1]],  # 88
            [-1, 1, "Conv", [256, 3, 1]],  # 89
            [-1, 1, "Conv", [256, 3, 1]],  # 90
            [[-1, -2, -3, -4, -5, -6], 1, "Concat", [1]],  # 91
            [-1, 1, "Conv", [512, 1, 1]],  # 92 P6
            [82, 1, "Conv", [768, 3, 1]],  # 93 P5 detect feature
            [92, 1, "Conv", [1024, 3, 1]],  # 94 P6 detect feature
            [[72, 93, 94], 1, "Detect", ["nc", "anchors"]],  # 95 Detect(P4,P5,P6)
        ]
    )
    cfg["head"] = head
    return cfg


def make_training_pruned(base, nc, width):
    cfg = make_pruned(base, nc, width)
    # Add auxiliary outputs for train_aux.py compatibility. The aux features match
    # P4/P5/P6 spatial sizes: layer 71, layer 59, and layer 47.
    cfg["head"] = deepcopy(cfg["head"][:-1])
    cfg["head"].extend(
        [
            [71, 1, "Conv", [640, 3, 1]],  # 95 P4 aux
            [59, 1, "Conv", [960, 3, 1]],  # 96 P5 aux
            [47, 1, "Conv", [1280, 3, 1]],  # 97 P6 aux
            [[72, 93, 94, 95, 96, 97], 1, "IAuxDetect", ["nc", "anchors"]],  # 98 Detect(P4,P5,P6)
        ]
    )
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Create P4/P5/P6 SIAV2 deploy cfg variants.")
    parser.add_argument("--base", default="cfg/deploy/yolov7-w6.yaml")
    parser.add_argument("--training-base", default="cfg/deploy/yolov7-w6.yaml")
    parser.add_argument("--out-dir", default="cfg/deploy")
    parser.add_argument("--training-out-dir", default="")
    parser.add_argument("--nc", type=int, default=16)
    parser.add_argument("--widths", nargs="+", type=float, default=[0.6, 0.55, 0.5])
    args = parser.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(args.training_base, "r", encoding="utf-8") as f:
        training_base = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    training_out_dir = Path(args.training_out_dir) if args.training_out_dir else None
    for width in args.widths:
        lite = make_lite(base, args.nc, width)
        pruned = make_pruned(base, args.nc, width)
        lite_path = out_dir / f"yolov7-l6-siav2-p4p6-lite-{tag(width)}.yaml"
        pruned_path = out_dir / f"yolov7-l6-siav2-p4p6-pruned-{tag(width)}.yaml"
        write_cfg(lite_path, lite)
        write_cfg(pruned_path, pruned)
        print(lite_path)
        print(pruned_path)
        if training_out_dir:
            training = make_training_pruned(training_base, args.nc, width)
            training_path = training_out_dir / f"yolov7-l6-siav2-p4p6-pruned-{tag(width)}.yaml"
            write_cfg(training_path, training)
            print(training_path)


if __name__ == "__main__":
    main()
