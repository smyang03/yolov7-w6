import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model
from utils.torch_utils import select_device


def main():
    parser = argparse.ArgumentParser(description="Build a YOLOv7 cfg and save an untrained random checkpoint.")
    parser.add_argument("--cfg", required=True, help="model yaml path")
    parser.add_argument("--out", required=True, help="output .pt path")
    parser.add_argument("--device", default="cpu", help="cpu or cuda device")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = select_device(args.device, batch_size=1)
    model = Model(args.cfg).to(device)
    model.names = [f"class_{i}" for i in range(model.yaml["nc"])]
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": -1,
        "best_fitness": 0.0,
        "model": model.cpu(),
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "wandb_id": None,
        "cfg": args.cfg,
        "nc": model.yaml["nc"],
        "seed": args.seed,
        "training": False,
    }
    torch.save(checkpoint, out_path)
    print(f"saved {out_path} cfg={args.cfg} nc={model.yaml['nc']} seed={args.seed}")


if __name__ == "__main__":
    main()
