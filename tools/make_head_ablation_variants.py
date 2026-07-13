import argparse
from copy import deepcopy
from pathlib import Path

import yaml


VARIANTS = {
    "dualhead": "DualDetect",
    "anchorfree": "AFDetect",
    "dual_anchorfree": "DualAFDetect",
}


def main():
    parser = argparse.ArgumentParser(description="Create latency-only SIAV2 head ablation cfgs.")
    parser.add_argument("--base", default="cfg/deploy/yolov7-l6-siav2.yaml")
    parser.add_argument("--out-dir", default="cfg/deploy")
    args = parser.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for suffix, module in VARIANTS.items():
        cfg = deepcopy(base)
        cfg["head"][-1][2] = module
        out = out_dir / f"yolov7-l6-siav2-{suffix}.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False, width=120)
        print(out)


if __name__ == "__main__":
    main()
