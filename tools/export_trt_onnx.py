import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import AFDetect, Detect, DualAFDetect, DualDetect, Model
from utils.torch_utils import select_device


def main():
    parser = argparse.ArgumentParser(description="Export a YOLOv7 cfg to ONNX for TensorRT profiling.")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--weights", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--img", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--constant-folding", action="store_true")
    parser.add_argument("--no-constant-folding", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = select_device(args.device, batch_size=1)
    model = Model(args.cfg).to(device)
    if args.weights:
        ckpt = torch.load(args.weights, map_location=device)
        src = (ckpt.get("ema") or ckpt.get("model")) if isinstance(ckpt, dict) else ckpt
        src_sd = src.float().state_dict() if hasattr(src, "state_dict") else src
        model_sd = model.state_dict()
        copied = 0
        for key, value in src_sd.items():
            if key in model_sd and model_sd[key].shape == value.shape:
                model_sd[key] = value.float()
                copied += 1
        model.load_state_dict(model_sd, strict=False)
        print(f"transferred {copied}/{len(model_sd)} items from {args.weights}")
    with torch.no_grad():
        model.fuse()
    model.eval()

    for module in model.modules():
        if isinstance(module, (Detect, AFDetect, DualDetect, DualAFDetect)):
            module.export = False
            module.concat = True
            module.include_nms = False
            module.end2end = False

    if args.half and device.type != "cpu":
        model.half()

    dtype = torch.float16 if args.half and device.type != "cpu" else torch.float32
    img = torch.zeros(1, 3, args.img, args.img, device=device, dtype=dtype)
    with torch.no_grad():
        y = model(img)
    print(f"dry_run_output_shape={tuple(y.shape) if hasattr(y, 'shape') else type(y)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        img,
        str(out),
        verbose=False,
        opset_version=args.opset,
        input_names=["images"],
        output_names=["output"],
        do_constant_folding=args.constant_folding and not args.no_constant_folding,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
