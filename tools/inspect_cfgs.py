import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model


def shape_list(tensors):
    return [list(t.shape) for t in tensors]


def main():
    parser = argparse.ArgumentParser(description="Build cfgs and run one no-grad forward pass.")
    parser.add_argument("--cfg", nargs="+", required=True)
    parser.add_argument("--img", type=int, default=1280)
    args = parser.parse_args()

    results = []
    for cfg in args.cfg:
        model = Model(cfg)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        x = torch.zeros(1, 3, args.img, args.img)
        with torch.no_grad():
            y = model(x)

        if isinstance(y, tuple):
            out_shape = list(y[0].shape)
            head_shapes = shape_list(y[1])
        else:
            out_shape = None
            head_shapes = []

        results.append(
            {
                "cfg": cfg,
                "nc": model.yaml["nc"],
                "width_multiple": model.yaml["width_multiple"],
                "params": params,
                "stride": model.model[-1].stride.tolist(),
                "anchors_shape": list(model.model[-1].anchors.shape),
                "out_shape": out_shape,
                "head_shapes": head_shapes,
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
