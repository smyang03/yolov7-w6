import argparse
import sys
from pathlib import Path
import yaml

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.yolo import Model
from utils.loss import ComputeLossAuxOTA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfg/training/yolov7-l6-siav2-p4small-coco80.yaml")
    parser.add_argument("--hyp", default="data/hyp.siav2-p4small-aux-reinforced.yaml")
    parser.add_argument("--img-size", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(args.hyp, encoding="utf-8") as f:
        hyp = yaml.safe_load(f)

    model = Model(args.cfg, ch=3, nc=80).to(device)
    model.hyp = hyp
    model.gr = 1.0
    model.train()

    imgs = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    targets = torch.tensor(
        [
            [0, 0, 0.50, 0.50, 0.015, 0.020],
            [0, 1, 0.25, 0.30, 0.050, 0.060],
            [1, 2, 0.75, 0.60, 0.020, 0.030],
            [1, 3, 0.40, 0.45, 0.150, 0.120],
        ],
        device=device,
    )

    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        pred = model(imgs)
        loss, items = ComputeLossAuxOTA(model)(pred, targets, imgs)

    print("loss", float(loss.detach().cpu()))
    print("items", [float(x) for x in items.detach().cpu()])
    print("small_obj_loss", hyp.get("small_obj_loss"), "min_dynamic_k", hyp.get("small_obj_min_dynamic_k"))
    print("pred_layers", len(pred), [tuple(p.shape) for p in pred])


if __name__ == "__main__":
    main()
