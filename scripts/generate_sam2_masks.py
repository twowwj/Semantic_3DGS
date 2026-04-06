import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate single-channel semantic GT masks with SAM2.")
    parser.add_argument("--input_dir", required=True, type=str, help="Directory containing training images.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save generated mask PNGs.")
    parser.add_argument("--sam2_config", required=True, type=str, help="SAM2 model config path.")
    parser.add_argument("--sam2_checkpoint", required=True, type=str, help="SAM2 checkpoint path.")
    parser.add_argument("--device", default="cuda", type=str, help="Inference device, usually cuda.")
    parser.add_argument("--points_per_side", default=32, type=int, help="SAM2 automatic mask generator sampling density.")
    parser.add_argument("--pred_iou_thresh", default=0.7, type=float, help="Minimum predicted IoU for generated masks.")
    parser.add_argument("--stability_score_thresh", default=0.85, type=float, help="Minimum stability score for generated masks.")
    parser.add_argument("--min_mask_region_area", default=100, type=int, help="Small connected regions below this area are removed by SAM2.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate masks even if output files already exist.")
    parser.add_argument("--disable_cudnn", action="store_true", help="Disable cuDNN and use PyTorch CUDA fallbacks if cuDNN initialization is broken.")
    return parser.parse_args()


def build_generator(args):
    try:
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise SystemExit(
            "Could not import SAM2. Install the official SAM2 package in the target environment first."
        ) from exc

    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as exc:
        raise SystemExit(
            "Could not import SAM2AutomaticMaskGenerator. "
            "Please make sure your SAM2 installation includes the automatic mask generator API."
        ) from exc

    model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=args.device)
    generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )
    return generator


def list_images(input_dir: Path):
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def generate_union_mask(generator, image_path: Path):
    image = np.array(Image.open(image_path).convert("RGB"))
    masks = generator.generate(image)
    union_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask_dict in masks:
        segmentation = mask_dict.get("segmentation")
        if segmentation is None:
            continue
        union_mask[segmentation.astype(bool)] = 255
    return union_mask, len(masks)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available but --device requested CUDA.")

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled for SAM2 mask generation.")

    generator = build_generator(args)
    image_paths = list_images(input_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")

    for image_path in image_paths:
        output_path = output_dir / f"{image_path.stem}.png"
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing mask: {output_path.name}")
            continue

        union_mask, num_masks = generate_union_mask(generator, image_path)
        cv2.imwrite(str(output_path), union_mask)
        print(f"Saved {output_path.name} with {num_masks} masks merged.")


if __name__ == "__main__":
    main()
