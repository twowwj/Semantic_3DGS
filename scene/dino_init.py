import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.graphics_utils import geom_transform_points


COMMON_FEATURE_DIMS = {8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536}


def resolve_dino_feature_dir(source_path, dino_feature_dir):
    if dino_feature_dir == "":
        return ""
    path = Path(dino_feature_dir)
    if not path.is_absolute():
        path = Path(source_path) / path
    return str(path)


def resolve_dino_feature_path(feature_dir, image_name):
    if feature_dir == "":
        return ""

    stem = Path(image_name).stem
    for ext in (".npy", ".npz", ".pt", ".pth"):
        candidate = Path(feature_dir) / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)
    return ""


def _npz_first_array(data):
    for key in ("features", "feature", "feat", "dino"):
        if key in data:
            return data[key]
    return data[data.files[0]]


def load_dino_feature(path, device):
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
        feature = torch.from_numpy(array)
    elif suffix == ".npz":
        with np.load(path) as data:
            feature = torch.from_numpy(_npz_first_array(data))
    elif suffix in {".pt", ".pth"}:
        feature = torch.load(path, map_location="cpu")
        if isinstance(feature, dict):
            for key in ("features", "feature", "feat", "dino"):
                if key in feature:
                    feature = feature[key]
                    break
            else:
                feature = next(iter(feature.values()))
        feature = torch.as_tensor(feature)
    else:
        raise ValueError(f"Unsupported DINO feature file extension: {path}")

    feature = feature.float()
    if feature.ndim == 4 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError(f"Expected 3D DINO feature map, got shape {tuple(feature.shape)} from {path}")

    first_dim, _, last_dim = feature.shape
    if last_dim in COMMON_FEATURE_DIMS and first_dim not in COMMON_FEATURE_DIMS:
        feature = feature.permute(2, 0, 1)
    elif last_dim in COMMON_FEATURE_DIMS and feature.shape[0] < feature.shape[-1]:
        feature = feature.permute(2, 0, 1)
    elif first_dim not in COMMON_FEATURE_DIMS and last_dim not in COMMON_FEATURE_DIMS:
        # Fall back to the common HWC export convention.
        feature = feature.permute(2, 0, 1)

    return feature.contiguous().to(device)


def project_points(points_xyz, camera):
    projected = geom_transform_points(points_xyz, camera.full_proj_transform)
    view_space = geom_transform_points(points_xyz, camera.world_view_transform)

    x_ndc = projected[:, 0]
    y_ndc = projected[:, 1]
    z_ndc = projected[:, 2]
    z_view = view_space[:, 2]

    x_pix = ((x_ndc + 1.0) * camera.image_width - 1.0) * 0.5
    y_pix = ((y_ndc + 1.0) * camera.image_height - 1.0) * 0.5

    valid = (
        torch.isfinite(x_pix)
        & torch.isfinite(y_pix)
        & torch.isfinite(z_ndc)
        & torch.isfinite(z_view)
        & (z_ndc > 0.0)
        & (z_view > 0.0)
        & (x_pix >= 0.0)
        & (x_pix < camera.image_width)
        & (y_pix >= 0.0)
        & (y_pix < camera.image_height)
    )
    return x_pix, y_pix, valid


def sample_feature_map(feature_map, x_pix, y_pix, image_width, image_height):
    if x_pix.numel() == 0:
        return feature_map.new_empty((0, feature_map.shape[0]))

    x_norm = (x_pix / max(image_width - 1, 1)) * 2.0 - 1.0
    y_norm = (y_pix / max(image_height - 1, 1)) * 2.0 - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        feature_map.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()


def reduce_features_to_dim(features, target_dim, max_samples):
    if features.shape[1] == target_dim:
        return F.normalize(features, dim=-1, eps=1e-6)

    valid = torch.isfinite(features).all(dim=-1) & (features.abs().sum(dim=-1) > 0)
    if valid.sum() == 0:
        return features.new_zeros((features.shape[0], target_dim))

    valid_features = features[valid]
    if valid_features.shape[0] > max_samples:
        sample_ids = torch.linspace(
            0,
            valid_features.shape[0] - 1,
            steps=max_samples,
            device=valid_features.device,
        ).long()
        fit_features = valid_features[sample_ids]
    else:
        fit_features = valid_features

    mean = fit_features.mean(dim=0, keepdim=True)
    centered_fit = fit_features - mean
    q = min(target_dim, centered_fit.shape[0], centered_fit.shape[1])
    if q == 0:
        return features.new_zeros((features.shape[0], target_dim))

    _, _, components = torch.pca_lowrank(centered_fit, q=q, center=False)
    reduced = (features - mean) @ components[:, :q]
    if q < target_dim:
        padding = reduced.new_zeros((reduced.shape[0], target_dim - q))
        reduced = torch.cat((reduced, padding), dim=-1)
    return F.normalize(reduced, dim=-1, eps=1e-6)


@torch.no_grad()
def build_dino_semantic_init(
    points_xyz,
    cameras,
    feature_dir,
    semantic_dim,
    max_views=8,
    chunk_size=65536,
    reduction_samples=20000,
):
    if feature_dir == "":
        return None

    selected_cameras = cameras if max_views <= 0 else cameras[:max_views]
    if len(selected_cameras) == 0:
        return None

    device = points_xyz.device
    feature_sum = None
    feature_count = torch.zeros((points_xyz.shape[0], 1), device=device)
    used_views = 0

    for camera in selected_cameras:
        feature_path = resolve_dino_feature_path(feature_dir, camera.image_name)
        if feature_path == "":
            print(f"[DINO init] Missing feature for {camera.image_name}, skipping.")
            continue

        feature_map = load_dino_feature(feature_path, device)
        if feature_sum is None:
            feature_sum = torch.zeros((points_xyz.shape[0], feature_map.shape[0]), device=device)
        elif feature_sum.shape[1] != feature_map.shape[0]:
            raise ValueError(
                f"DINO feature channel mismatch: expected {feature_sum.shape[1]}, got {feature_map.shape[0]} at {feature_path}"
            )

        for start in range(0, points_xyz.shape[0], chunk_size):
            end = min(start + chunk_size, points_xyz.shape[0])
            x_pix, y_pix, valid = project_points(points_xyz[start:end], camera)
            if not valid.any():
                continue
            valid_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            sampled = sample_feature_map(
                feature_map,
                x_pix[valid],
                y_pix[valid],
                camera.image_width,
                camera.image_height,
            )
            global_ids = valid_ids + start
            feature_sum.index_add_(0, global_ids, sampled)
            feature_count.index_add_(0, global_ids, torch.ones((global_ids.shape[0], 1), device=device))

        used_views += 1
        print(f"[DINO init] Aggregated {camera.image_name} from {feature_path}")

    if feature_sum is None or used_views == 0:
        print("[DINO init] No DINO features were found; using zero semantic initialization.")
        return None

    valid = feature_count.squeeze(-1) > 0
    averaged = feature_sum / feature_count.clamp_min(1.0)
    averaged[~valid] = 0.0
    reduced = reduce_features_to_dim(averaged, semantic_dim, reduction_samples)
    reduced[~valid] = 0.0

    coverage = valid.float().mean().item()
    print(
        f"[DINO init] Initialized semantic embeddings from {used_views} views; "
        f"point coverage={coverage:.2%}; feature_dim={feature_sum.shape[1]} -> semantic_dim={semantic_dim}."
    )
    return reduced
