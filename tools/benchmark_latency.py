import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model
from utils.torch_utils import select_device


class GpuMonitor:
    def __init__(self, device_index=0, interval=0.2):
        self.device_index = device_index
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        query = [
            "nvidia-smi",
            f"--id={self.device_index}",
            "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(query, text=True, stderr=subprocess.DEVNULL).strip()
                if out:
                    parts = [p.strip() for p in out.split(",")]
                    self.samples.append(
                        {
                            "timestamp": parts[0],
                            "util_gpu": int(parts[1]),
                            "mem_used_mib": int(parts[2]),
                            "mem_total_mib": int(parts[3]),
                        }
                    )
            except Exception:
                pass
            self._stop.wait(self.interval)

    @property
    def max_util(self):
        return max((s["util_gpu"] for s in self.samples), default=None)

    @property
    def max_mem_used(self):
        return max((s["mem_used_mib"] for s in self.samples), default=None)


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] * (c - k) + values[c] * (k - f)


def cfg_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {"nc": data.get("nc"), "width_multiple": data.get("width_multiple"), "depth_multiple": data.get("depth_multiple")}


def benchmark_one(cfg, device, img_size, warmup, repeat, sleep_s, half):
    model = Model(cfg).to(device)
    with torch.no_grad():
        model.fuse()
    model.eval()
    if half and device.type != "cpu":
        model.half()

    params = sum(p.numel() for p in model.parameters())
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    x = x.half() if half and device.type != "cpu" else x.float()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device.type != "cpu":
                torch.cuda.synchronize()
            if sleep_s:
                time.sleep(sleep_s)

        times_ms = []
        for _ in range(repeat):
            if device.type != "cpu":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(x)
                end.record()
                torch.cuda.synchronize()
                times_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = model(x)
                times_ms.append((time.perf_counter() - t0) * 1000.0)
            if sleep_s:
                time.sleep(sleep_s)

    return {
        "cfg": cfg,
        "img_size": img_size,
        "device": str(device),
        "half": bool(half and device.type != "cpu"),
        "params": params,
        "times_ms": times_ms,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "p90_ms": percentile(times_ms, 90),
        **cfg_meta(cfg),
    }


def main():
    parser = argparse.ArgumentParser(description="Forward-only latency benchmark for pre-training model selection.")
    parser.add_argument("--cfg", nargs="+", required=True, help="cfg paths to benchmark")
    parser.add_argument("--device", default="0", help="cpu or CUDA device id")
    parser.add_argument("--img", type=int, default=1280, help="square input size")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--sleep", type=float, default=0.75, help="sleep between forwards to keep sampled GPU util low")
    parser.add_argument("--half", action="store_true", help="use FP16 on CUDA")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-cap", type=int, default=30)
    parser.add_argument("--json-out", default="", help="write full JSON result")
    parser.add_argument("--csv-out", default="", help="append CSV summary")
    args = parser.parse_args()

    device = select_device(args.device, batch_size=1)
    if device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    results = []
    with GpuMonitor(args.gpu_index) if device.type != "cpu" else nullcontext() as mon:
        for cfg in args.cfg:
            result = benchmark_one(cfg, device, args.img, args.warmup, args.repeat, args.sleep, args.half)
            results.append(result)
        monitor = {
            "max_gpu_util": mon.max_util if device.type != "cpu" else None,
            "max_mem_used_mib": mon.max_mem_used if device.type != "cpu" else None,
            "sample_count": len(mon.samples) if device.type != "cpu" else 0,
            "gpu_cap": args.gpu_cap,
            "cap_violation": bool(device.type != "cpu" and mon.max_util is not None and mon.max_util >= args.gpu_cap),
        }

    baseline = results[0]["median_ms"] if results else None
    for r in results:
        r["ratio_vs_first"] = (r["median_ms"] / baseline) if baseline else None
        r["speed_gate_pass_vs_first"] = bool(baseline and r["median_ms"] <= baseline * 0.5)

    payload = {"monitor": monitor, "results": results}
    print(json.dumps(payload, indent=2))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        exists = out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "cfg",
                    "img_size",
                    "device",
                    "half",
                    "nc",
                    "width_multiple",
                    "params",
                    "median_ms",
                    "mean_ms",
                    "p90_ms",
                    "ratio_vs_first",
                    "speed_gate_pass_vs_first",
                    "max_gpu_util",
                    "cap_violation",
                ],
            )
            if not exists:
                writer.writeheader()
            for r in results:
                row = dict(r)
                row.pop("times_ms", None)
                row["max_gpu_util"] = monitor["max_gpu_util"]
                row["cap_violation"] = monitor["cap_violation"]
                writer.writerow({k: row.get(k) for k in writer.fieldnames})


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    main()
