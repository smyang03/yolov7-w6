import argparse
import csv
import json
import re
from pathlib import Path


MODEL_RE = re.compile(r"/model\.(\d+)[/.]")


def layer_index(name):
    match = MODEL_RE.search(name)
    return int(match.group(1)) if match else None


def parse_log(path):
    rows = []
    total = None
    path = Path(path)
    text = ""
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le"):
        try:
            candidate = path.read_text(encoding=encoding, errors="ignore")
        except UnicodeError:
            continue
        if "[I]" in candidate:
            text = candidate
            break
    if not text:
        text = path.read_text(encoding="utf-8", errors="ignore")

    for line in text.splitlines():
        if "[I]" not in line:
            continue
        payload = line.split("[I]", 1)[1].strip()
        parts = payload.split()
        if len(parts) < 5:
            continue
        try:
            time_ms = float(parts[0])
            avg_ms = float(parts[1])
            median_ms = float(parts[2])
            pct = float(parts[3])
        except ValueError:
            continue
        row = {
            "time_ms": time_ms,
            "avg_ms": avg_ms,
            "median_ms": median_ms,
            "pct": pct,
            "layer": " ".join(parts[4:]).strip(),
        }
        if row["layer"] == "Total":
            total = row
        else:
            row["idx"] = layer_index(row["layer"])
            rows.append(row)
    max_idx = max((r["idx"] for r in rows if r["idx"] is not None), default=None)
    for row in rows:
        idx = row["idx"]
        if idx is None:
            row["stage"] = "other"
        elif max_idx is not None and idx == max_idx:
            row["stage"] = "detect_decode"
        elif idx <= 46:
            row["stage"] = "backbone"
        else:
            row["stage"] = "neck"
    return rows, total


def summarize(path):
    rows, total = parse_log(path)
    by_stage = {}
    by_idx = {}
    for row in rows:
        by_stage[row["stage"]] = by_stage.get(row["stage"], 0.0) + row["avg_ms"]
        if row["idx"] is not None:
            by_idx[row["idx"]] = by_idx.get(row["idx"], 0.0) + row["avg_ms"]
    return {
        "log": str(path),
        "name": Path(path).stem.replace(".trtexec", ""),
        "total_avg_ms": total["avg_ms"] if total else None,
        "total_median_ms": total["median_ms"] if total else None,
        "by_stage": by_stage,
        "top_indices": sorted(by_idx.items(), key=lambda x: x[1], reverse=True)[:15],
        "top_layers": sorted(rows, key=lambda x: x["avg_ms"], reverse=True)[:25],
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize TensorRT trtexec dumpProfile logs.")
    parser.add_argument("--log", nargs="+", required=True)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--csv-out", default="")
    args = parser.parse_args()

    summaries = [summarize(Path(p)) for p in args.log]
    payload = json.dumps(summaries, indent=2)
    print(payload)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "total_avg_ms",
                    "total_median_ms",
                    "backbone_ms",
                    "neck_ms",
                    "detect_decode_ms",
                    "other_ms",
                ],
            )
            writer.writeheader()
            for item in summaries:
                stages = item["by_stage"]
                writer.writerow(
                    {
                        "name": item["name"],
                        "total_avg_ms": item["total_avg_ms"],
                        "total_median_ms": item["total_median_ms"],
                        "backbone_ms": stages.get("backbone", 0.0),
                        "neck_ms": stages.get("neck", 0.0),
                        "detect_decode_ms": stages.get("detect_decode", 0.0),
                        "other_ms": stages.get("other", 0.0),
                    }
                )


if __name__ == "__main__":
    main()
