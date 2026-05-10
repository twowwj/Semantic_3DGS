#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MODEL_PATCH_SIZES = {
    "dinov2_vits14": 14,
    "dinov2_vitb14": 14,
    "dinov2_vitl14": 14,
    "dinov2_vitg14": 14,
}


def list_images(image_dir):
    return sorted(path for path in Path(image_dir).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def round_to_patch_size(value, patch_size):
    return max(patch_size, int(round(value / patch_size)) * patch_size)


def resize_for_dino(image, max_long_side, patch_size):
    width, height = image.size
    if max_long_side > 0:
        scale = min(1.0, max_long_side / max(width, height))
        width = int(round(width * scale))
        height = int(round(height * scale))

    width = round_to_patch_size(width, patch_size)
    height = round_to_patch_size(height, patch_size)
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.BICUBIC)


def image_to_tensor(image, device):
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


@torch.no_grad()
def extract_feature_map(model, image_tensor, patch_size):
    features = model.forward_features(image_tensor)
    if isinstance(features, dict):
        if "x_norm_patchtokens" in features:
            patch_tokens = features["x_norm_patchtokens"]
        elif "x_prenorm" in features:
            patch_tokens = features["x_prenorm"][:, 1:]
        else:
            raise KeyError(f"Could not find patch tokens in DINO output keys: {list(features.keys())}")
    else:
        patch_tokens = features[:, 1:] if features.ndim == 3 else features

    _, _, height, width = image_tensor.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    patch_tokens = patch_tokens.reshape(1, grid_h, grid_w, -1)
    return patch_tokens.squeeze(0).float().cpu().numpy()


def save_feature(path, feature, compressed):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path.with_suffix(".npz"), features=feature)
    else:
        np.save(path.with_suffix(".npy"), feature)


def main():
    parser = ArgumentParser(description="Extract dense DINOv2 patch features for Gaussian semantic initialization.")
    parser.add_argument("--image_dir", required=True, type=str, help="Directory containing RGB input images.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory where per-image DINO feature maps are saved.")
    parser.add_argument("--model", default="dinov2_vits14", choices=sorted(MODEL_PATCH_SIZES), help="DINOv2 torch.hub model name.")
    parser.add_argument("--device", default="cuda", type=str, help="Torch device used for feature extraction.")
    parser.add_argument("--max_long_side", default=840, type=int, help="Resize longest image side before extraction. Use 0 to keep original size.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files.")
    parser.add_argument("--compressed", action="store_true", help="Save .npz instead of .npy.")
    parser.add_argument("--disable_cudnn", action="store_true", help="Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED in fragile containers.")
    args = parser.parse_args()

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled for DINO feature extraction.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    patch_size = MODEL_PATCH_SIZES[args.model]

    print(f"Loading {args.model} on {device}")
    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model.eval().to(device)

    image_paths = list_images(args.image_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {args.image_dir}")

    output_dir = Path(args.output_dir)
    suffix = ".npz" if args.compressed else ".npy"

    for image_path in tqdm(image_paths, desc="Extracting DINO features"):
        output_path = output_dir / f"{image_path.stem}{suffix}"
        if output_path.exists() and not args.overwrite:
            continue

        image = Image.open(image_path).convert("RGB")
        image = resize_for_dino(image, args.max_long_side, patch_size)
        image_tensor = image_to_tensor(image, device)
        feature = extract_feature_map(model, image_tensor, patch_size)
        save_feature(output_path, feature, args.compressed)

    print(f"Saved DINO features to {output_dir}")


if __name__ == "__main__":
    main()
