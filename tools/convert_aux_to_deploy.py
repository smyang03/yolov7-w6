import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.yolo import Model


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a YOLOv7 IAuxDetect checkpoint to a deploy Detect checkpoint.")
    parser.add_argument("--weights", required=True, help="training checkpoint with IAuxDetect")
    parser.add_argument("--deploy-cfg", required=True, help="deploy cfg with Detect")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def checkpoint_model(ckpt):
    if isinstance(ckpt, dict):
        model = ckpt.get("ema") or ckpt.get("model")
    else:
        model = ckpt
    if model is None:
        raise ValueError("checkpoint does not contain model or ema")
    return deepcopy(model).float().cpu().eval()


def fuse_implicit_conv(conv, ia=None, im=None):
    weight = conv.weight.detach().float().clone()
    bias = conv.bias.detach().float().clone()
    if ia is not None:
        implicit_add = ia.implicit.detach().float().reshape(-1)
        bias = bias + torch.matmul(weight.reshape(weight.shape[0], -1), implicit_add).reshape(-1)
    if im is not None:
        implicit_mul = im.implicit.detach().float().reshape(-1)
        weight = weight * implicit_mul.reshape(-1, 1, 1, 1)
        bias = bias * implicit_mul
    return weight, bias


def copy_matching(src_model, deploy_model):
    src_sd = src_model.state_dict()
    deploy_sd = deploy_model.state_dict()
    copied = 0
    for key, value in src_sd.items():
        if key in deploy_sd and deploy_sd[key].shape == value.shape:
            deploy_sd[key] = value.float()
            copied += 1
    deploy_model.load_state_dict(deploy_sd, strict=False)
    return copied


def copy_detect_head(src_model, deploy_model):
    src_det = src_model.model[-1]
    deploy_det = deploy_model.model[-1]
    copied = 0
    if not hasattr(src_det, "m") or not hasattr(deploy_det, "m"):
        return copied

    n = min(len(src_det.m), len(deploy_det.m))
    for i in range(n):
        ia = src_det.ia[i] if hasattr(src_det, "ia") else None
        im = src_det.im[i] if hasattr(src_det, "im") else None
        weight, bias = fuse_implicit_conv(src_det.m[i], ia, im)
        if deploy_det.m[i].weight.shape == weight.shape and deploy_det.m[i].bias.shape == bias.shape:
            deploy_det.m[i].weight.data.copy_(weight)
            deploy_det.m[i].bias.data.copy_(bias)
            copied += 1
    return copied


def main():
    args = parse_args()
    ckpt = torch.load(args.weights, map_location="cpu")
    src_model = checkpoint_model(ckpt)
    deploy_model = Model(args.deploy_cfg).float().cpu().eval()

    copied_backbone = copy_matching(src_model, deploy_model)
    copied_head = copy_detect_head(src_model, deploy_model)

    if hasattr(src_model, "names"):
        deploy_model.names = src_model.names
    if hasattr(src_model, "nc"):
        deploy_model.nc = src_model.nc

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": ckpt.get("epoch", -1) if isinstance(ckpt, dict) else -1,
        "best_fitness": ckpt.get("best_fitness", 0.0) if isinstance(ckpt, dict) else 0.0,
        "model": deploy_model.half(),
        "ema": None,
        "updates": ckpt.get("updates", 0) if isinstance(ckpt, dict) else 0,
        "optimizer": None,
        "wandb_id": None,
    }
    torch.save(payload, out)
    print(f"saved {out}")
    print(f"copied_matching={copied_backbone} copied_detect_heads={copied_head}")


if __name__ == "__main__":
    main()
