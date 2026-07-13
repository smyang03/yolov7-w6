import argparse
from pathlib import Path

import yaml


DEFAULT_WIDTHS = (0.75, 0.67, 0.50, 0.40)


def variant_name(width):
    return f"yolov7-l6-siav2-w{int(round(width * 1000)):03d}.yaml"


def main():
    parser = argparse.ArgumentParser(description="Create SIAV2 pre-training cfg candidates from YOLOv7-W6.")
    parser.add_argument("--base", default="cfg/training/yolov7-w6.yaml", help="source W6 cfg")
    parser.add_argument("--out-dir", default="cfg/training", help="output directory")
    parser.add_argument("--nc", type=int, default=16, help="class count")
    parser.add_argument("--widths", nargs="*", type=float, default=list(DEFAULT_WIDTHS), help="width multiples to emit")
    args = parser.parse_args()

    base_path = Path(args.base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with base_path.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    written = []
    for width in args.widths:
        cfg = dict(base)
        cfg["nc"] = args.nc
        cfg["width_multiple"] = float(width)
        out_path = out_dir / variant_name(width)
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False, width=120)
        written.append(out_path)

    print("created:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
