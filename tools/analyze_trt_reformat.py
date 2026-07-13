import argparse
import csv
import json
import re
from pathlib import Path


PROFILE_RE = re.compile(
    r"\[I\]\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(.+)$"
)


def read_text(path):
    data = Path(path).read_bytes()
    for encoding in ("utf-16", "utf-8-sig", "utf-8", "cp949"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="ignore")


def parse_profile(path):
    rows = []
    for line in read_text(path).splitlines():
        match = PROFILE_RE.search(line)
        if not match:
            continue
        total_ms, avg_ms, median_ms, pct, layer = match.groups()
        rows.append(
            {
                "total_ms": float(total_ms),
                "avg_ms": float(avg_ms),
                "median_ms": float(median_ms),
                "pct": float(pct),
                "layer": layer.strip(),
            }
        )
    return rows


def summarize(path):
    rows = parse_profile(path)
    copy_rows = [
        row
        for row in rows
        if "copy" in row["layer"].lower() or "reformat" in row["layer"].lower()
    ]
    total_avg = sum(row["avg_ms"] for row in rows)
    copy_avg = sum(row["avg_ms"] for row in copy_rows)
    return {
        "log": str(path),
        "name": Path(path).stem.replace(".trtexec", ""),
        "layer_count": len(rows),
        "copy_reformat_count": len(copy_rows),
        "total_profile_avg_ms": round(total_avg, 6),
        "copy_reformat_avg_ms": round(copy_avg, 6),
        "copy_reformat_pct_of_profile": round((copy_avg / total_avg * 100.0), 3) if total_avg else 0.0,
        "top_copy_reformat": sorted(copy_rows, key=lambda row: row["avg_ms"], reverse=True)[:20],
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize TensorRT copy/reformat profile rows.")
    parser.add_argument("--log", nargs="+", required=True)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--csv-out", default="")
    args = parser.parse_args()

    summaries = [summarize(path) for path in args.log]
    print(json.dumps(summaries, indent=2))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.csv_out).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "layer_count",
                    "copy_reformat_count",
                    "total_profile_avg_ms",
                    "copy_reformat_avg_ms",
                    "copy_reformat_pct_of_profile",
                ],
            )
            writer.writeheader()
            for row in summaries:
                writer.writerow({key: row[key] for key in writer.fieldnames})


if __name__ == "__main__":
    main()
