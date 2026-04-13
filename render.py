#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


SEMANTIC_COLORS = torch.tensor([
    [0.129, 0.588, 0.953],  # background: blue
    [1.000, 0.549, 0.000],  # foreground: orange
], dtype=torch.float32)


def crop_for_train_test_exp(image, enabled):
    if enabled:
        return image[..., image.shape[-1] // 2:]
    return image


def colorize_binary_map(label_map):
    palette = SEMANTIC_COLORS.to(label_map.device)
    colored = palette[label_map.long()]
    return colored.permute(2, 0, 1)


def colorize_heatmap(values):
    values = values.float()
    values = values - values.min()
    denom = values.max().clamp_min(1e-6)
    values = values / denom
    return torch.stack((values, torch.zeros_like(values), 1.0 - values), dim=0)


def get_output_stem(view, idx):
    image_name = getattr(view, "image_name", "")
    if image_name:
        return os.path.splitext(os.path.basename(image_name))[0]
    return f"{idx:05d}"


def build_semantic_visuals(semantic_image, prototypes, semantic_gt=None):
    features = semantic_image.permute(1, 2, 0)
    normalized_features = F.normalize(features, dim=-1, eps=1e-6)
    normalized_prototypes = F.normalize(prototypes, dim=-1, eps=1e-6)
    logits = normalized_features @ normalized_prototypes.t()
    prediction = logits.argmax(dim=-1)

    visuals = {
        "semantic_pred": colorize_binary_map(prediction),
    }

    if logits.shape[-1] >= 2:
        visuals["semantic_bg_score"] = colorize_heatmap(logits[..., 0])
        visuals["semantic_fg_score"] = colorize_heatmap(logits[..., 1])

    if semantic_gt is not None:
        gt_binary = (semantic_gt.squeeze(0) >= 0.5).long()
        visuals["semantic_gt"] = colorize_binary_map(gt_binary)

    return visuals


def make_overview_grid(rendering, gt, semantic_visuals):
    tiles = [
        rendering.cpu(),
        gt.cpu(),
        semantic_visuals["semantic_pred"].cpu(),
        semantic_visuals.get("semantic_fg_score", torch.zeros_like(rendering)).cpu(),
        semantic_visuals.get("semantic_bg_score", torch.zeros_like(rendering)).cpu(),
    ]

    if "semantic_gt" in semantic_visuals:
        tiles.append(semantic_visuals["semantic_gt"].cpu())

    return torchvision.utils.make_grid(tiles, nrow=3, padding=8)


def maybe_log_render_to_wandb(wandb_run, split_name, idx, view_name, rendering, gt, semantic_visuals):
    if not wandb_run or idx >= 3:
        return

    log_payload = {
        f"render/{split_name}/{idx:03d}_{view_name}/render": wandb.Image(rendering.permute(1, 2, 0).cpu().numpy()),
        f"render/{split_name}/{idx:03d}_{view_name}/gt": wandb.Image(gt.permute(1, 2, 0).cpu().numpy()),
        f"render/{split_name}/{idx:03d}_{view_name}/semantic_pred": wandb.Image(semantic_visuals["semantic_pred"].permute(1, 2, 0).cpu().numpy()),
    }
    if "semantic_gt" in semantic_visuals:
        log_payload[f"render/{split_name}/{idx:03d}_{view_name}/semantic_gt"] = wandb.Image(semantic_visuals["semantic_gt"].permute(1, 2, 0).cpu().numpy())
    if "semantic_fg_score" in semantic_visuals:
        log_payload[f"render/{split_name}/{idx:03d}_{view_name}/semantic_fg_score"] = wandb.Image(semantic_visuals["semantic_fg_score"].permute(1, 2, 0).cpu().numpy())
    if "semantic_bg_score" in semantic_visuals:
        log_payload[f"render/{split_name}/{idx:03d}_{view_name}/semantic_bg_score"] = wandb.Image(semantic_visuals["semantic_bg_score"].permute(1, 2, 0).cpu().numpy())
    wandb_run.log(log_payload)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, wandb_run=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    semantic_path = os.path.join(model_path, name, "ours_{}".format(iteration), "semantic")
    overview_path = os.path.join(model_path, name, "ours_{}".format(iteration), "semantic_overview")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(semantic_path, exist_ok=True)
    makedirs(overview_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output_stem = get_output_stem(view, idx)
        render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        semantic_image = render_pkg.get("semantic")

        rendering = crop_for_train_test_exp(rendering, train_test_exp)
        gt = crop_for_train_test_exp(gt, train_test_exp)

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{output_stem}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{output_stem}.png"))

        if semantic_image is None or semantic_image.numel() == 0:
            continue

        semantic_image = crop_for_train_test_exp(semantic_image, train_test_exp)
        semantic_gt = view.semantic_map
        if semantic_gt is not None:
            semantic_gt = crop_for_train_test_exp(semantic_gt, train_test_exp)

        semantic_visuals = build_semantic_visuals(
            semantic_image,
            gaussians.get_semantic_prototypes.detach(),
            semantic_gt=semantic_gt,
        )

        for suffix, image in semantic_visuals.items():
            torchvision.utils.save_image(image, os.path.join(semantic_path, f"{output_stem}_{suffix}.png"))

        overview = make_overview_grid(rendering, gt, semantic_visuals)
        torchvision.utils.save_image(overview, os.path.join(overview_path, f"{output_stem}.png"))
        maybe_log_render_to_wandb(wandb_run, name, idx, view.image_name, rendering, gt, semantic_visuals)

def init_wandb(args, job_type):
    if not args.use_wandb:
        return None
    if not WANDB_FOUND:
        print("wandb requested but not installed: not logging to wandb")
        return None

    wandb_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name if args.wandb_run_name else None,
        "entity": args.wandb_entity if args.wandb_entity else None,
        "config": vars(args),
        "dir": args.model_path,
        "job_type": job_type,
        "reinit": True,
    }
    return wandb.init(**{k: v for k, v in wandb_kwargs.items() if v is not None})


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, wandb_run=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, semantic_dim=dataset.semantic_dim)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, wandb_run)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, wandb_run)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    wandb_run = init_wandb(args, "render")

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, wandb_run)
    if wandb_run:
        wandb_run.finish()
