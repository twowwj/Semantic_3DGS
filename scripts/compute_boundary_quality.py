import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize foreground boundaries and floaters by comparing baseline and semantic renders."
    )
    parser.add_argument("--baseline", required=True, type=str, help="Directory containing baseline rendered images.")
    parser.add_argument("--semantic", required=True, type=str, help="Directory containing semantic-loss rendered images.")
    parser.add_argument("--gt", required=True, type=str, help="Directory containing ground-truth images.")
    parser.add_argument("--mask", required=True, type=str, help="Directory containing semantic GT masks.")
    parser.add_argument("--output", required=True, type=str, help="Directory to save comparison panels.")
    parser.add_argument("--crop_size", default=192, type=int, help="Square crop size used for local boundary inspection.")
    parser.add_argument("--num_crops", default=3, type=int, help="Number of automatically selected boundary crops per image.")
    parser.add_argument("--max_images", default=0, type=int, help="Limit the number of processed images. 0 means all.")
    parser.add_argument("--error_scale", default=4.0, type=float, help="Amplification factor for error-map visualization.")
    parser.add_argument(
        "--crop_mode",
        default="improvement",
        choices=["improvement", "boundary", "mixed"],
        help="How to select local crops: semantic improvement regions, mask boundaries, or a weighted mix of both.",
    )
    parser.add_argument("--top_k", default=0, type=int, help="If > 0, only export the top-K images with the highest semantic improvement score.")
    return parser.parse_args()


def list_images(directory: Path):
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_rgb(path: Path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path: Path):
    mask = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return mask


def normalize_error_map(render: np.ndarray, gt: np.ndarray, scale: float):
    error = np.abs(render.astype(np.float32) - gt.astype(np.float32)).mean(axis=2)
    error = np.clip(error * scale, 0, 255).astype(np.uint8)
    return np.stack([error, error, error], axis=2)


def compute_error_intensity(render: np.ndarray, gt: np.ndarray):
    return np.abs(render.astype(np.float32) - gt.astype(np.float32)).mean(axis=2)


def compute_boundary_map(mask: np.ndarray):
    binary = mask > 127
    padded = np.pad(binary.astype(np.uint8), 1, mode="edge")
    center = padded[1:-1, 1:-1]
    neighbors = [
        padded[:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[:-2, :-2],
        padded[:-2, 2:],
        padded[2:, :-2],
        padded[2:, 2:],
    ]
    boundary = np.zeros_like(center, dtype=bool)
    for neighbor in neighbors:
        boundary |= center != neighbor
    return boundary.astype(np.uint8)


def pick_crop_centers(boundary_map: np.ndarray, crop_size: int, num_crops: int):
    ys, xs = np.nonzero(boundary_map)
    height, width = boundary_map.shape
    if len(xs) == 0:
        return [(width // 2, height // 2)]

    scores = []
    radius = max(crop_size // 2, 1)
    integral = boundary_map.astype(np.int32).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(x0, y0, x1, y1):
        total = integral[y1 - 1, x1 - 1]
        if x0 > 0:
            total -= integral[y1 - 1, x0 - 1]
        if y0 > 0:
            total -= integral[y0 - 1, x1 - 1]
        if x0 > 0 and y0 > 0:
            total += integral[y0 - 1, x0 - 1]
        return total

    for x, y in zip(xs, ys):
        x0 = max(0, x - radius)
        y0 = max(0, y - radius)
        x1 = min(width, x0 + crop_size)
        y1 = min(height, y0 + crop_size)
        x0 = max(0, x1 - crop_size)
        y0 = max(0, y1 - crop_size)
        score = rect_sum(x0, y0, x1, y1)
        scores.append((int(score), x, y))

    scores.sort(reverse=True)
    selected = []
    min_dist_sq = max((crop_size // 2) ** 2, 1)
    for _, x, y in scores:
        if all((x - sx) ** 2 + (y - sy) ** 2 >= min_dist_sq for sx, sy in selected):
            selected.append((x, y))
        if len(selected) >= num_crops:
            break

    if not selected:
        selected = [(width // 2, height // 2)]
    return selected


def pick_crop_centers_from_score(score_map: np.ndarray, crop_size: int, num_crops: int):
    height, width = score_map.shape
    radius = max(crop_size // 2, 1)
    integral = score_map.astype(np.float32).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(x0, y0, x1, y1):
        total = integral[y1 - 1, x1 - 1]
        if x0 > 0:
            total -= integral[y1 - 1, x0 - 1]
        if y0 > 0:
            total -= integral[y0 - 1, x1 - 1]
        if x0 > 0 and y0 > 0:
            total += integral[y0 - 1, x0 - 1]
        return float(total)

    candidates = []
    step = max(crop_size // 6, 8)
    for y in range(0, height, step):
        for x in range(0, width, step):
            x0 = max(0, x - radius)
            y0 = max(0, y - radius)
            x1 = min(width, x0 + crop_size)
            y1 = min(height, y0 + crop_size)
            x0 = max(0, x1 - crop_size)
            y0 = max(0, y1 - crop_size)
            candidates.append((rect_sum(x0, y0, x1, y1), x, y))

    candidates.sort(reverse=True)
    selected = []
    min_dist_sq = max((crop_size // 2) ** 2, 1)
    for score, x, y in candidates:
        if score <= 0:
            continue
        if all((x - sx) ** 2 + (y - sy) ** 2 >= min_dist_sq for sx, sy in selected):
            selected.append((x, y))
        if len(selected) >= num_crops:
            break

    if not selected:
        selected = [(width // 2, height // 2)]
    return selected


def extract_crop(image: np.ndarray, center, crop_size: int):
    x, y = center
    height, width = image.shape[:2]
    half = crop_size // 2
    x0 = max(0, x - half)
    y0 = max(0, y - half)
    x1 = min(width, x0 + crop_size)
    y1 = min(height, y0 + crop_size)
    x0 = max(0, x1 - crop_size)
    y0 = max(0, y1 - crop_size)
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def resize_crop(image: np.ndarray, size: int):
    return np.array(Image.fromarray(image).resize((size, size), Image.Resampling.NEAREST))


def add_title(image: np.ndarray, title: str):
    pil_img = Image.fromarray(image)
    canvas = Image.new("RGB", (pil_img.width, pil_img.height + 26), color=(255, 255, 255))
    canvas.paste(pil_img, (0, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))
    return np.array(canvas)


def stack_h(images):
    return np.concatenate(images, axis=1)


def stack_v(images):
    max_width = max(image.shape[1] for image in images)
    padded = []
    for image in images:
        if image.shape[1] == max_width:
            padded.append(image)
            continue
        pad_width = max_width - image.shape[1]
        pad = np.full((image.shape[0], pad_width, image.shape[2]), 255, dtype=image.dtype)
        padded.append(np.concatenate([image, pad], axis=1))
    return np.concatenate(padded, axis=0)


def ensure_same_names(baseline_dir: Path, semantic_dir: Path, gt_dir: Path):
    baseline_names = {p.name for p in list_images(baseline_dir)}
    semantic_names = {p.name for p in list_images(semantic_dir)}
    gt_names = {p.name for p in list_images(gt_dir)}
    common = sorted(baseline_names & semantic_names & gt_names)
    return common


def find_mask_path(mask_dir: Path, image_name: str):
    stem = Path(image_name).stem
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
        path = mask_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def draw_crop_boxes(image: np.ndarray, boxes, color):
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    for box in boxes:
        draw.rectangle(box, outline=color, width=3)
    return np.array(pil_img)


def build_score_map(args, boundary_map, baseline_error_raw, semantic_error_raw):
    improvement = np.clip(baseline_error_raw - semantic_error_raw, 0.0, None)
    boundary_float = boundary_map.astype(np.float32)

    if args.crop_mode == "boundary":
        return boundary_float
    if args.crop_mode == "mixed":
        return improvement + 32.0 * boundary_float
    return improvement


def compute_improvement_score(boundary_map: np.ndarray, baseline_error_raw: np.ndarray, semantic_error_raw: np.ndarray):
    boundary_weight = 1.0 + 4.0 * boundary_map.astype(np.float32)
    delta = (baseline_error_raw - semantic_error_raw) * boundary_weight
    return float(delta.mean()), float(np.clip(delta, 0.0, None).sum())


def main():
    args = parse_args()
    baseline_dir = Path(args.baseline)
    semantic_dir = Path(args.semantic)
    gt_dir = Path(args.gt)
    mask_dir = Path(args.mask)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_names = ensure_same_names(baseline_dir, semantic_dir, gt_dir)
    if args.max_images > 0:
        image_names = image_names[: args.max_images]

    if not image_names:
        raise SystemExit("No common rendered images found across baseline, semantic, and GT directories.")

    ranked_items = []
    for image_name in image_names:
        mask_path = find_mask_path(mask_dir, image_name)
        if mask_path is None:
            print(f"Skipping {image_name}: no matching mask found.")
            continue

        baseline = load_rgb(baseline_dir / image_name)
        semantic = load_rgb(semantic_dir / image_name)
        gt = load_rgb(gt_dir / image_name)
        mask = load_mask(mask_path)

        baseline_error_raw = compute_error_intensity(baseline, gt)
        semantic_error_raw = compute_error_intensity(semantic, gt)
        baseline_error = normalize_error_map(baseline, gt, args.error_scale)
        semantic_error = normalize_error_map(semantic, gt, args.error_scale)
        boundary_map = compute_boundary_map(mask)
        mean_improvement, positive_improvement = compute_improvement_score(boundary_map, baseline_error_raw, semantic_error_raw)
        ranked_items.append(
            {
                "image_name": image_name,
                "mask_path": str(mask_path),
                "mean_improvement": mean_improvement,
                "positive_improvement": positive_improvement,
            }
        )

    if not ranked_items:
        raise SystemExit("No images could be paired with masks.")

    ranked_items.sort(key=lambda item: item["positive_improvement"], reverse=True)
    with open(output_dir / "improvement_ranking.json", "w") as f:
        json.dump(ranked_items, f, indent=2)

    if args.top_k > 0:
        selected_items = ranked_items[: args.top_k]
    else:
        selected_items = ranked_items

    for item in selected_items:
        image_name = item["image_name"]
        mask_path = Path(item["mask_path"])

        baseline = load_rgb(baseline_dir / image_name)
        semantic = load_rgb(semantic_dir / image_name)
        gt = load_rgb(gt_dir / image_name)
        mask = load_mask(mask_path)

        baseline_error_raw = compute_error_intensity(baseline, gt)
        semantic_error_raw = compute_error_intensity(semantic, gt)
        baseline_error = normalize_error_map(baseline, gt, args.error_scale)
        semantic_error = normalize_error_map(semantic, gt, args.error_scale)
        boundary_map = compute_boundary_map(mask)
        score_map = build_score_map(args, boundary_map, baseline_error_raw, semantic_error_raw)
        if args.crop_mode == "boundary":
            crop_centers = pick_crop_centers(boundary_map, args.crop_size, args.num_crops)
        else:
            crop_centers = pick_crop_centers_from_score(score_map, args.crop_size, args.num_crops)

        crop_boxes = []
        crop_panels = []
        crop_vis_size = 256
        for crop_idx, center in enumerate(crop_centers, start=1):
            gt_crop, box = extract_crop(gt, center, args.crop_size)
            baseline_crop, _ = extract_crop(baseline, center, args.crop_size)
            semantic_crop, _ = extract_crop(semantic, center, args.crop_size)
            baseline_error_crop, _ = extract_crop(baseline_error, center, args.crop_size)
            semantic_error_crop, _ = extract_crop(semantic_error, center, args.crop_size)
            mask_crop, _ = extract_crop(np.stack([mask, mask, mask], axis=2), center, args.crop_size)
            crop_boxes.append(box)

            crop_row = stack_h([
                add_title(resize_crop(gt_crop, crop_vis_size), f"Crop {crop_idx} GT"),
                add_title(resize_crop(baseline_crop, crop_vis_size), f"Crop {crop_idx} Baseline"),
                add_title(resize_crop(semantic_crop, crop_vis_size), f"Crop {crop_idx} Semantic"),
                add_title(resize_crop(baseline_error_crop, crop_vis_size), f"Crop {crop_idx} Base Error"),
                add_title(resize_crop(semantic_error_crop, crop_vis_size), f"Crop {crop_idx} Sem Error"),
                add_title(resize_crop(mask_crop, crop_vis_size), f"Crop {crop_idx} Mask"),
            ])
            crop_panels.append(crop_row)

        overview_gt = draw_crop_boxes(gt, crop_boxes, (255, 0, 0))
        overview_baseline = draw_crop_boxes(baseline, crop_boxes, (255, 0, 0))
        overview_semantic = draw_crop_boxes(semantic, crop_boxes, (255, 0, 0))
        overview_mask = draw_crop_boxes(np.stack([mask, mask, mask], axis=2), crop_boxes, (255, 0, 0))

        overview_row = stack_h([
            add_title(overview_gt, f"GT ({args.crop_mode})"),
            add_title(overview_baseline, "Baseline"),
            add_title(overview_semantic, "Semantic"),
            add_title(baseline_error, "Baseline Error"),
            add_title(semantic_error, "Semantic Error"),
            add_title(overview_mask, "Mask"),
        ])

        full_panel = stack_v([overview_row] + crop_panels)
        output_path = output_dir / image_name
        Image.fromarray(full_panel).save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
