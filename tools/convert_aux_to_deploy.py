import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.yolo import Model
from utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a YOLOv7 IAuxDetect checkpoint to a deploy Detect checkpoint.")
    parser.add_argument("--weights", required=True, help="training checkpoint with IAuxDetect")
    parser.add_argument("--deploy-cfg", required=True, help="deploy cfg with Detect")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-strict", action="store_false", dest="strict", help="allow incomplete non-head copy")
    parser.add_argument("--verify-forward", action="store_true", help="compare source lead output with deploy output")
    parser.add_argument("--verify-img", type=int, default=640, help="image size for --verify-forward")
    parser.add_argument("--verify-atol", type=float, default=1e-4)
    parser.add_argument("--verify-rtol", type=float, default=1e-4)
    parser.set_defaults(strict=True)
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


def copy_matching(src_model, deploy_model, strict=True):
    src_sd = src_model.state_dict()
    deploy_sd = deploy_model.state_dict()
    deploy_head_prefix = f"model.{len(deploy_model.model) - 1}."
    copied = []
    missing = []
    mismatched = []
    for key, value in deploy_sd.items():
        if key.startswith(deploy_head_prefix):
            continue
        src_value = src_sd.get(key)
        if src_value is None:
            missing.append(key)
        elif src_value.shape != value.shape:
            mismatched.append((key, tuple(src_value.shape), tuple(value.shape)))
        else:
            deploy_sd[key] = src_value.float()
            copied.append(key)
    if strict and (missing or mismatched):
        detail = []
        if missing:
            detail.append(f"missing={missing[:8]}")
        if mismatched:
            detail.append(f"mismatched={mismatched[:8]}")
        raise RuntimeError("incomplete aux->deploy non-head copy: " + " ".join(detail))
    deploy_model.load_state_dict(deploy_sd, strict=False)
    return len(copied), missing, mismatched


def copy_detect_head(src_model, deploy_model, strict=True):
    src_det = src_model.model[-1]
    deploy_det = deploy_model.model[-1]
    copied = 0
    if not hasattr(src_det, "m") or not hasattr(deploy_det, "m"):
        if strict:
            raise RuntimeError("source/deploy model does not expose a compatible Detect head")
        return copied

    if strict and len(src_det.m) < deploy_det.nl:
        raise RuntimeError(f"source head has {len(src_det.m)} lead heads, deploy requires {deploy_det.nl}")
    if strict and len(deploy_det.m) != deploy_det.nl:
        raise RuntimeError(f"deploy head count {len(deploy_det.m)} does not match nl={deploy_det.nl}")

    for i in range(deploy_det.nl):
        ia = src_det.ia[i] if hasattr(src_det, "ia") else None
        im = src_det.im[i] if hasattr(src_det, "im") else None
        weight, bias = fuse_implicit_conv(src_det.m[i], ia, im)
        if deploy_det.m[i].weight.shape != weight.shape or deploy_det.m[i].bias.shape != bias.shape:
            message = (
                f"head {i} shape mismatch: "
                f"src_weight={tuple(weight.shape)} deploy_weight={tuple(deploy_det.m[i].weight.shape)} "
                f"src_bias={tuple(bias.shape)} deploy_bias={tuple(deploy_det.m[i].bias.shape)}"
            )
            if strict:
                raise RuntimeError(message)
            print(f"warning: {message}")
            continue
        deploy_det.m[i].weight.data.copy_(weight)
        deploy_det.m[i].bias.data.copy_(bias)
        copied += 1
    if strict and copied != deploy_det.nl:
        raise RuntimeError(f"copied {copied} detect heads, expected {deploy_det.nl}")
    return copied


def output_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"cannot extract prediction tensor from output type {type(output)}")


def verify_forward(src_model, deploy_model, device, img_size, atol, rtol):
    src_model = src_model.to(device).float().eval()
    deploy_model = deploy_model.to(device).float().eval()
    img = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        src_out = output_tensor(src_model(img))
        deploy_out = output_tensor(deploy_model(img))
    if src_out.shape != deploy_out.shape:
        raise RuntimeError(f"forward shape mismatch: source={tuple(src_out.shape)} deploy={tuple(deploy_out.shape)}")
    max_abs = (src_out - deploy_out).abs().max().item()
    if not torch.allclose(src_out, deploy_out, atol=atol, rtol=rtol):
        raise RuntimeError(f"forward parity failed: max_abs={max_abs:.6g} atol={atol} rtol={rtol}")
    return max_abs


def main():
    args = parse_args()
    ckpt = torch.load(args.weights, map_location="cpu")
    src_model = checkpoint_model(ckpt)
    deploy_model = Model(args.deploy_cfg).float().cpu().eval()

    copied_backbone, missing_backbone, mismatched_backbone = copy_matching(src_model, deploy_model, strict=args.strict)
    copied_head = copy_detect_head(src_model, deploy_model, strict=args.strict)

    max_abs = None
    if args.verify_forward:
        device = select_device(args.device, batch_size=1)
        max_abs = verify_forward(src_model, deploy_model, device, args.verify_img, args.verify_atol, args.verify_rtol)
        deploy_model = deploy_model.cpu().eval()

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
    if missing_backbone or mismatched_backbone:
        print(f"non_strict_missing={len(missing_backbone)} non_strict_mismatched={len(mismatched_backbone)}")
    if max_abs is not None:
        print(f"forward_parity_max_abs={max_abs:.6g}")


if __name__ == "__main__":
    main()
