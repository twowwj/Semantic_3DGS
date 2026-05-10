from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch

from arguments import ModelParams, get_combined_args
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.graphics_utils import geom_transform_points

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False


def pick_view(views, view_name=None, view_index=None):
    if view_name:
        stem = Path(view_name).stem
        for view in views:
            if Path(view.image_name).stem == stem or view.image_name == view_name:
                return view
        raise ValueError(f"Could not find view '{view_name}' in the selected split.")

    if view_index is None:
        view_index = 0

    if view_index < 0 or view_index >= len(views):
        raise IndexError(f"view_index={view_index} is out of range for {len(views)} views.")
    return views[view_index]


def project_points(points_xyz, camera):
    projected = geom_transform_points(points_xyz, camera.full_proj_transform)
    view_space = geom_transform_points(points_xyz, camera.world_view_transform)

    x_ndc = projected[:, 0]
    y_ndc = projected[:, 1]
    z_ndc = projected[:, 2]
    z_view = view_space[:, 2]

    x_pix = ((x_ndc + 1.0) * camera.image_width - 1.0) * 0.5
    # geom_transform_points() in this codebase already follows the rasterizer's
    # screen-space convention, so applying an extra (1 - y) here flips the
    # projected centers vertically relative to the rendered image.
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
    return x_pix[valid], y_pix[valid], z_view[valid], valid


def filter_surface_points(x_pix, y_pix, z_view, depth_image, rel_tol, abs_tol):
    if x_pix.numel() == 0:
        return x_pix, y_pix, z_view

    depth_map = depth_image.squeeze(0).detach()
    h, w = depth_map.shape
    x_int = x_pix.round().long().clamp_(0, w - 1)
    y_int = y_pix.round().long().clamp_(0, h - 1)

    expected_invdepth = depth_map[y_int, x_int]
    point_invdepth = 1.0 / z_view.clamp_min(1e-6)
    tolerance = torch.maximum(
        expected_invdepth.abs() * rel_tol,
        torch.full_like(expected_invdepth, abs_tol),
    )
    near_surface = (expected_invdepth > 0) & ((point_invdepth - expected_invdepth).abs() <= tolerance)
    return x_pix[near_surface], y_pix[near_surface], z_view[near_surface]


def sample_points(x_pix, y_pix, max_points):
    if x_pix.numel() <= max_points:
        return x_pix, y_pix

    generator = torch.Generator(device=x_pix.device)
    generator.manual_seed(0)
    keep = torch.randperm(x_pix.numel(), generator=generator, device=x_pix.device)[:max_points]
    return x_pix[keep], y_pix[keep]


def draw_points(base_image, x_pix, y_pix, color, radius):
    canvas = base_image.copy()
    h, w = canvas.shape[:2]
    x_int = x_pix.round().long().clamp_(0, w - 1)
    y_int = y_pix.round().long().clamp_(0, h - 1)
    xy = torch.stack((x_int, y_int), dim=1).cpu().numpy()
    for x, y in xy:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


def build_density_map(x_pix, y_pix, width, height):
    density = np.zeros((height, width), dtype=np.float32)
    x_int = x_pix.round().long().clamp_(0, width - 1)
    y_int = y_pix.round().long().clamp_(0, height - 1)
    xy = torch.stack((x_int, y_int), dim=1).cpu().numpy()
    for x, y in xy:
        if 0 <= x < width and 0 <= y < height:
            density[y, x] += 1.0

    density = cv2.GaussianBlur(density, (0, 0), sigmaX=5.0, sigmaY=5.0)
    if density.max() > 0:
        density = density / density.max()
    heatmap = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return heatmap


def compute_error_intensity(render_bgr, gt_bgr):
    render = render_bgr.astype(np.float32)
    gt = gt_bgr.astype(np.float32)
    return np.abs(render - gt).mean(axis=2)


def normalize_error_map(render_bgr, gt_bgr, scale=4.0):
    error = compute_error_intensity(render_bgr, gt_bgr)
    error = np.clip(error * scale, 0, 255).astype(np.uint8)
    return np.stack([error, error, error], axis=2)


def build_semantic_pred_vis(semantic_tensor):
    if semantic_tensor is None or semantic_tensor.numel() == 0:
        return None

    pred = (semantic_tensor[:1] >= 0.5).float().squeeze(0).detach().cpu().numpy()
    vis = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    vis[pred > 0.5] = np.array([0, 165, 255], dtype=np.uint8)
    vis[pred <= 0.5] = np.array([255, 144, 30], dtype=np.uint8)
    return vis


def pick_crop_centers_from_score(score_map, crop_size, num_crops):
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


def extract_crop(image, center, crop_size):
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


def draw_crop_boxes(image, boxes, color):
    canvas = image.copy()
    for box in boxes:
        x0, y0, x1, y1 = box
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, 2, lineType=cv2.LINE_AA)
    return canvas


def get_boundary_band(semantic_map, width, height, kernel_size):
    if semantic_map is None:
        return None

    mask = semantic_map.squeeze(0).detach().cpu().numpy()
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = (mask >= 0.5).astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = ((dilated - eroded) > 0).astype(np.uint8)
    return mask, boundary


def overlay_boundary(image, boundary):
    canvas = image.copy()
    if boundary is None:
        return canvas
    canvas[boundary > 0] = np.array([0, 255, 255], dtype=np.uint8)
    return canvas


def to_uint8_image(tensor_image):
    image = tensor_image.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def add_title(image, title):
    header_h = 36
    canvas = np.full((image.shape[0] + header_h, image.shape[1], 3), 235, dtype=np.uint8)
    canvas[header_h:] = image
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    return canvas


def resize_to_height(image, target_height):
    if image.shape[0] == target_height:
        return image
    scale = target_height / image.shape[0]
    target_width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)


def stack_h(images):
    target_height = max(image.shape[0] for image in images)
    resized = [resize_to_height(image, target_height) for image in images]
    return np.concatenate(resized, axis=1)


def stack_v(images):
    max_width = max(image.shape[1] for image in images)
    padded = []
    for image in images:
        if image.shape[1] == max_width:
            padded.append(image)
            continue
        pad_width = max_width - image.shape[1]
        pad = np.full((image.shape[0], pad_width, image.shape[2]), 235, dtype=image.dtype)
        padded.append(np.concatenate([image, pad], axis=1))
    return np.concatenate(padded, axis=0)


def init_wandb(args, dataset, job_type="gaussian-centers"):
    if not getattr(args, "use_wandb", False):
        return None
    if not WANDB_FOUND:
        print("wandb requested but not installed: not logging to wandb")
        return None

    run_name = args.wandb_run_name if args.wandb_run_name else f"{Path(dataset.model_path).name}-{job_type}"
    wandb_kwargs = {
        "project": args.wandb_project,
        "name": run_name,
        "entity": args.wandb_entity if args.wandb_entity else None,
        "config": vars(args),
        "dir": dataset.model_path,
        "job_type": job_type,
        "reinit": True,
    }
    return wandb.init(**{k: v for k, v in wandb_kwargs.items() if v is not None})


def main():
    parser = ArgumentParser(description="Visualize projected Gaussian centers for a reconstructed 3DGS model.")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--view_name", default="", type=str)
    parser.add_argument("--view_index", default=None, type=int)
    parser.add_argument("--kernel_size", default=9, type=int)
    parser.add_argument("--max_points", default=30000, type=int)
    parser.add_argument("--point_radius", default=1, type=int)
    parser.add_argument("--depth_rel_tol", default=0.1, type=float)
    parser.add_argument("--depth_abs_tol", default=0.01, type=float)
    parser.add_argument("--background_source", choices=["render", "gt"], default="render")
    parser.add_argument("--crop_size", default=192, type=int)
    parser.add_argument("--num_crops", default=1, type=int)
    parser.add_argument("--error_scale", default=4.0, type=float)
    parser.add_argument("--output", default="", type=str)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    args.view_name = getattr(args, "view_name", "")
    args.view_index = getattr(args, "view_index", None)
    args.kernel_size = getattr(args, "kernel_size", 9)
    args.max_points = getattr(args, "max_points", 30000)
    args.point_radius = getattr(args, "point_radius", 1)
    args.depth_rel_tol = getattr(args, "depth_rel_tol", 0.1)
    args.depth_abs_tol = getattr(args, "depth_abs_tol", 0.01)
    args.background_source = getattr(args, "background_source", "render")
    args.crop_size = getattr(args, "crop_size", 192)
    args.num_crops = getattr(args, "num_crops", 1)
    args.error_scale = getattr(args, "error_scale", 4.0)
    args.output = getattr(args, "output", "")
    args.split = getattr(args, "split", "train")

    safe_state(args.quiet)

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, semantic_dim=dataset.semantic_dim)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    wandb_run = init_wandb(args, dataset)

    try:
        views = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
        view = pick_view(views, view_name=args.view_name, view_index=args.view_index)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipe = type("VisualizationPipe", (), {
            "debug": False,
            "antialiasing": False,
            "compute_cov3D_python": False,
            "convert_SHs_python": False,
        })()

        with torch.no_grad():
            render_pkg = render(view, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp, separate_sh=False)

        visibility_filter = render_pkg["visibility_filter"].squeeze(-1)
        if visibility_filter.numel() == 0:
            raise RuntimeError("No visible gaussians were found for the selected view.")

        xyz = gaussians.get_xyz.detach()[visibility_filter]
        x_pix, y_pix, z_view, valid = project_points(xyz, view)
        screen_filtered_count = x_pix.numel()
        x_pix, y_pix, z_view = filter_surface_points(
            x_pix,
            y_pix,
            z_view,
            render_pkg["depth"],
            rel_tol=args.depth_rel_tol,
            abs_tol=args.depth_abs_tol,
        )
        x_pix, y_pix = sample_points(x_pix, y_pix, args.max_points)

        gt_rgb = to_uint8_image(view.original_image[:3])
        render_rgb = to_uint8_image(render_pkg["render"])
        rgb = render_rgb if args.background_source == "render" else gt_rgb
        h, w = rgb.shape[:2]
        error_map = normalize_error_map(render_rgb, gt_rgb, scale=args.error_scale)
        semantic_pred_vis = build_semantic_pred_vis(render_pkg["semantic"])

        semantic_info = get_boundary_band(view.semantic_map, w, h, args.kernel_size)
        if semantic_info is None:
            mask_vis = np.full_like(rgb, 255)
            boundary = None
        else:
            mask, boundary = semantic_info
            mask_vis = np.zeros_like(rgb)
            mask_vis[mask > 0] = np.array([0, 165, 255], dtype=np.uint8)
            mask_vis[mask == 0] = np.array([255, 144, 30], dtype=np.uint8)

        centers_overlay = draw_points(rgb, x_pix, y_pix, color=(0, 255, 0), radius=args.point_radius)
        boundary_overlay = overlay_boundary(centers_overlay, boundary)

        if boundary is not None:
            xi = x_pix.round().long().clamp_(0, w - 1).cpu().numpy()
            yi = y_pix.round().long().clamp_(0, h - 1).cpu().numpy()
            inside_boundary = boundary[yi, xi] > 0
            boundary_points = int(inside_boundary.sum())
            boundary_only = draw_points(rgb, x_pix[inside_boundary], y_pix[inside_boundary], color=(0, 0, 255), radius=max(args.point_radius, 2))
            boundary_ratio = boundary_points / max(len(xi), 1)
        else:
            boundary_points = 0
            boundary_ratio = 0.0
            boundary_only = rgb.copy()

        density = build_density_map(x_pix, y_pix, w, h)

        crop_boxes = []
        if boundary is not None:
            score_map = compute_error_intensity(render_rgb, gt_rgb) * (1.0 + 4.0 * boundary.astype(np.float32))
            crop_centers = pick_crop_centers_from_score(score_map, args.crop_size, args.num_crops)
            crop_center = crop_centers[0]
            render_crop, crop_box = extract_crop(render_rgb, crop_center, args.crop_size)
            gt_crop, _ = extract_crop(gt_rgb, crop_center, args.crop_size)
            error_crop, _ = extract_crop(error_map, crop_center, args.crop_size)
            if semantic_pred_vis is not None:
                pred_crop, _ = extract_crop(semantic_pred_vis, crop_center, args.crop_size)
            else:
                pred_crop = np.full_like(render_crop, 255)
            crop_boxes.append(crop_box)
        else:
            render_crop = gt_rgb.copy()
            gt_crop = gt_rgb.copy()
            error_crop = error_map.copy()
            pred_crop = semantic_pred_vis if semantic_pred_vis is not None else np.full_like(gt_rgb, 255)

        overview_with_box = draw_crop_boxes(boundary_overlay, crop_boxes, color=(255, 255, 255)) if crop_boxes else boundary_overlay

        if semantic_pred_vis is None:
            semantic_pred_vis = np.full_like(rgb, 255)

        tiles = [
            add_title(rgb, f"Overlay Background ({args.background_source}): {view.image_name}"),
            add_title(mask_vis, "Semantic GT"),
            add_title(semantic_pred_vis, "Semantic Pred"),
            add_title(overview_with_box, "Centers + Boundary + Crop"),
            add_title(boundary_only, f"Boundary Centers ({boundary_points})"),
            add_title(density, "Center Density"),
            add_title(error_map, "Render-vs-GT Error"),
            add_title(render_crop, "Render Crop"),
            add_title(gt_crop, "GT Crop"),
            add_title(error_crop, "Error Crop"),
            add_title(pred_crop, "Semantic Pred Crop"),
            add_title(gt_rgb, "GT RGB"),
        ]

        row1 = stack_h(tiles[:3])
        row2 = stack_h(tiles[3:6])
        row3 = stack_h(tiles[6:9])
        row4 = stack_h(tiles[9:12])
        panel = stack_v([row1, row2, row3, row4])

        default_output = Path(dataset.model_path) / "center_visualizations" / f"{args.split}_{Path(view.image_name).stem}_centers.png"
        output_path = Path(args.output) if args.output else default_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), panel)

        print(f"Saved visualization to: {output_path}")
        print(f"View: {view.image_name}")
        print(f"Visible gaussians from renderer: {visibility_filter.numel()}")
        print(f"Visible projected centers after screen filtering: {screen_filtered_count}")
        print(f"Surface-consistent centers after depth filtering: {x_pix.numel()}")
        if boundary is not None:
            print(f"Boundary-band centers: {boundary_points}")
            print(f"Boundary-band ratio: {boundary_ratio:.4f}")
        else:
            print("No semantic map found for this view; boundary statistics were skipped.")

        if wandb_run:
            view_key = f"{args.split}/{Path(view.image_name).stem}"
            wandb_run.log({
                f"gaussian_centers/{view_key}/panel": wandb.Image(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)),
                f"gaussian_centers/{view_key}/rgb": wandb.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)),
                f"gaussian_centers/{view_key}/render_rgb": wandb.Image(cv2.cvtColor(render_rgb, cv2.COLOR_BGR2RGB)),
                f"gaussian_centers/{view_key}/gt_rgb": wandb.Image(cv2.cvtColor(gt_rgb, cv2.COLOR_BGR2RGB)),
                f"gaussian_centers/{view_key}/semantic_gt": wandb.Image(cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)),
                f"gaussian_centers/{view_key}/semantic_pred": wandb.Image(cv2.cvtColor(semantic_pred_vis, cv2.COLOR_BGR2RGB)),
                f"gaussian_centers/{view_key}/centers_boundary": wandb.Image(cv2.cvtColor(boundary_overlay, cv2.COLOR_BGR2RGB)),
                f"gaussian_centers/{view_key}/boundary_centers": wandb.Image(cv2.cvtColor(boundary_only, cv2.COLOR_BGR2RGB)),
                f"gaussian_centers/{view_key}/density": wandb.Image(cv2.cvtColor(density, cv2.COLOR_BGR2RGB)),
                f"gaussian_centers/{view_key}/error_map": wandb.Image(cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)),
                f"gaussian_centers/{view_key}/render_crop": wandb.Image(cv2.cvtColor(render_crop, cv2.COLOR_BGR2RGB)),
                f"gaussian_centers/{view_key}/gt_crop": wandb.Image(cv2.cvtColor(gt_crop, cv2.COLOR_BGR2RGB)),
                f"gaussian_centers/{view_key}/error_crop": wandb.Image(cv2.cvtColor(error_crop, cv2.COLOR_BGR2RGB)),
                f"gaussian_centers/{view_key}/visible_centers": int(screen_filtered_count),
                f"gaussian_centers/{view_key}/surface_centers": int(x_pix.numel()),
                f"gaussian_centers/{view_key}/boundary_centers_count": int(boundary_points),
                f"gaussian_centers/{view_key}/boundary_centers_ratio": float(boundary_ratio),
            })
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
