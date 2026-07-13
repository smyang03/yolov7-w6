import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def resolve_path(path, base):
    path = Path(path)
    if path.is_absolute():
        return path
    base_path = (base / path).resolve()
    if base_path.exists():
        return base_path
    cwd_path = (Path.cwd() / path).resolve()
    return cwd_path if cwd_path.exists() else base_path


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def image_to_label(path):
    parts = list(path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return path.with_suffix(".txt")


def expand_images(source, base):
    if isinstance(source, (list, tuple)):
        images = []
        for item in source:
            images.extend(expand_images(item, base))
        return images

    path = resolve_path(str(source), base)
    if path.is_file() and path.suffix.lower() == ".txt":
        images = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            img = Path(line)
            if img.is_absolute():
                images.append(img)
            else:
                from_list = (path.parent / img).resolve()
                from_cwd = (Path.cwd() / img).resolve()
                images.append(from_list if from_list.exists() else from_cwd)
        return images
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if path.is_file() and path.suffix.lower() in IMG_EXTS:
        return [path]
    return []


def read_label(path):
    rows = []
    if not path.exists():
        return rows, False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        rows.append((cls, x, y, w, h))
    return rows, True


def load_anchors(cfg_path):
    if not cfg_path:
        return []
    cfg = read_yaml(cfg_path)
    anchors = cfg.get("anchors", [])
    flat = []
    for layer in anchors:
        vals = list(layer)
        flat.extend((float(vals[i]), float(vals[i + 1])) for i in range(0, len(vals), 2))
    return flat


def anchor_ratio(wh, anchors):
    if not anchors:
        return None
    w, h = wh
    best = None
    for aw, ah in anchors:
        ratio = max(w / aw, aw / max(w, 1e-9), h / ah, ah / max(h, 1e-9))
        best = ratio if best is None else min(best, ratio)
    return best


def pct(value, total):
    return round(value / total * 100.0, 3) if total else 0.0


def analyze_split(name, source, base, imgsz, small_thresholds, anchors, anchor_t):
    images = expand_images(source, base)
    class_counts = Counter()
    side_bins = Counter()
    area_bins = Counter()
    anchor_hits = 0
    anchor_ratios = []
    missing_labels = 0
    empty_labels = 0
    boxes = []

    for img in images:
        label_path = image_to_label(img)
        labels, exists = read_label(label_path)
        if not exists:
            missing_labels += 1
        if exists and not labels:
            empty_labels += 1
        for cls, _x, _y, w, h in labels:
            max_side_px = max(w, h) * imgsz
            area_px = w * h * imgsz * imgsz
            class_counts[cls] += 1
            boxes.append((cls, max_side_px, area_px))
            for threshold in small_thresholds:
                if max_side_px <= threshold:
                    side_bins[str(threshold)] += 1
            for threshold in (32 * 32, 64 * 64, 96 * 96, 128 * 128):
                if area_px <= threshold:
                    area_bins[str(threshold)] += 1
            ratio = anchor_ratio((w * imgsz, h * imgsz), anchors)
            if ratio is not None:
                anchor_ratios.append(ratio)
                if ratio < anchor_t:
                    anchor_hits += 1

    box_count = len(boxes)
    summary = {
        "split": name,
        "images": len(images),
        "boxes": box_count,
        "missing_label_files": missing_labels,
        "empty_label_files": empty_labels,
        "boxes_per_image": round(box_count / len(images), 4) if images else 0.0,
        "class_counts": dict(sorted(class_counts.items())),
        "max_side_px_le": {k: {"count": v, "pct": pct(v, box_count)} for k, v in sorted(side_bins.items(), key=lambda x: float(x[0]))},
        "area_px_le": {k: {"count": v, "pct": pct(v, box_count)} for k, v in sorted(area_bins.items(), key=lambda x: float(x[0]))},
    }
    if anchor_ratios:
        sorted_ratios = sorted(anchor_ratios)
        summary["anchor_fit"] = {
            "anchor_t": anchor_t,
            "best_possible_recall_pct": pct(anchor_hits, box_count),
            "ratio_p50": round(sorted_ratios[int(0.50 * (len(sorted_ratios) - 1))], 4),
            "ratio_p90": round(sorted_ratios[int(0.90 * (len(sorted_ratios) - 1))], 4),
            "ratio_p95": round(sorted_ratios[int(0.95 * (len(sorted_ratios) - 1))], 4),
        }
    return summary


def write_markdown(path, payload):
    lines = ["# SIAV2 Dataset EDA", ""]
    lines.append(f"data: `{payload['data']}`")
    lines.append(f"imgsz: `{payload['imgsz']}`")
    lines.append("")
    for split in payload["splits"]:
        lines.append(f"## {split['split']}")
        lines.append("")
        lines.append(f"- images: `{split['images']}`")
        lines.append(f"- boxes: `{split['boxes']}`")
        lines.append(f"- boxes/image: `{split['boxes_per_image']}`")
        lines.append(f"- missing label files: `{split['missing_label_files']}`")
        lines.append(f"- empty label files: `{split['empty_label_files']}`")
        if split.get("anchor_fit"):
            fit = split["anchor_fit"]
            lines.append(f"- anchor BPR@{fit['anchor_t']}: `{fit['best_possible_recall_pct']}%`")
            lines.append(f"- anchor ratio p50/p90/p95: `{fit['ratio_p50']}` / `{fit['ratio_p90']}` / `{fit['ratio_p95']}`")
        lines.append("")
        lines.append("| max side px <= | count | pct |")
        lines.append("|---:|---:|---:|")
        for threshold, item in split["max_side_px_le"].items():
            lines.append(f"| {threshold} | {item['count']} | {item['pct']} |")
        lines.append("")
        lines.append("| class | boxes |")
        lines.append("|---:|---:|")
        for cls, count in split["class_counts"].items():
            lines.append(f"| {cls} | {count} |")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize SIAV2 YOLO label distribution and anchor fit.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--cfg", default="", help="optional cfg yaml for anchor fit")
    parser.add_argument("--anchor-t", type=float, default=4.0)
    parser.add_argument("--small-thresholds", nargs="*", type=float, default=[32, 64, 96, 128])
    parser.add_argument("--json-out", default="runs/siav2/eda/siav2_dataset_eda.json")
    parser.add_argument("--md-out", default="runs/siav2/eda/siav2_dataset_eda.md")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    data = read_yaml(data_path)
    base = data_path.parent
    anchors = load_anchors(args.cfg)
    payload = {
        "data": str(data_path),
        "imgsz": args.imgsz,
        "cfg": args.cfg,
        "anchors": anchors,
        "splits": [],
    }
    for split_name in ("train", "val", "test"):
        if split_name in data and data[split_name]:
            payload["splits"].append(
                analyze_split(split_name, data[split_name], base, args.imgsz, args.small_thresholds, anchors, args.anchor_t)
            )

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(md_out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
