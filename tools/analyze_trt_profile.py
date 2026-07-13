import argparse
import csv
import json
import re
from pathlib import Path


MODEL_RE = re.compile(r"/model\.(\d+)[/.]")


def load_profile(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    rows = [r for r in rows if isinstance(r, dict) and "name" in r]
    return rows


def layer_index(name):
    match = MODEL_RE.search(name)
    return int(match.group(1)) if match else None


def stage(idx):
    if idx is None:
        return "other"
    if idx <= 46:
        return "backbone"
    if idx <= 113:
        return "neck"
    return "detect_decode"


def summarize(path):
    rows = load_profile(path)
    total = next((r for r in rows if r["name"] == "Total"), None)
    layers = [r for r in rows if r["name"] != "Total" and r.get("averageMs", 0) > 0]
    for r in layers:
        r["idx"] = layer_index(r["name"])
        r["stage"] = stage(r["idx"])

    by_stage = {}
    by_idx = {}
    for r in layers:
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0.0) + float(r.get("averageMs", 0.0))
        if r["idx"] is not None:
            by_idx[r["idx"]] = by_idx.get(r["idx"], 0.0) + float(r.get("averageMs", 0.0))

    return {
        "path": str(path),
        "total_average_ms": total.get("averageMs") if total else None,
        "total_median_ms": total.get("medianMs") if total else None,
        "by_stage": by_stage,
        "top_indices": sorted(by_idx.items(), key=lambda x: x[1], reverse=True)[:15],
        "top_layers": sorted(layers, key=lambda x: float(x.get("averageMs", 0.0)), reverse=True)[:25],
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize TensorRT trtexec profile JSON.")
    parser.add_argument("--profile", nargs="+", required=True)
    parser.add_argument("--csv-out", default="")
    args = parser.parse_args()

    summaries = [summarize(Path(p)) for p in args.profile]
    print(json.dumps(summaries, indent=2))

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["profile", "total_average_ms", "total_median_ms", "backbone_ms", "neck_ms", "detect_decode_ms", "other_ms"],
            )
            writer.writeheader()
            for s in summaries:
                stages = s["by_stage"]
                writer.writerow(
                    {
                        "profile": Path(s["path"]).stem.replace(".profile", ""),
                        "total_average_ms": s["total_average_ms"],
                        "total_median_ms": s["total_median_ms"],
                        "backbone_ms": stages.get("backbone", 0.0),
                        "neck_ms": stages.get("neck", 0.0),
                        "detect_decode_ms": stages.get("detect_decode", 0.0),
                        "other_ms": stages.get("other", 0.0),
                    }
                )


if __name__ == "__main__":
    main()
