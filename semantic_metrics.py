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

import json
import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False


def crop_for_train_test_exp(image, enabled):
    if enabled:
        return image[..., image.shape[-1] // 2:]
    return image


def predict_binary_semantics(semantic_image, prototypes, temperature):
    features = semantic_image.permute(1, 2, 0)
    normalized_features = F.normalize(features, dim=-1, eps=1e-6)
    normalized_prototypes = F.normalize(prototypes, dim=-1, eps=1e-6)
    logits = normalized_features @ normalized_prototypes.t()
    logits = logits / max(temperature, 1e-6)
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    foreground_scores = probabilities[..., 1] if probabilities.shape[-1] > 1 else probabilities[..., 0]
    return predictions, foreground_scores


def compute_binary_metrics(prediction, target):
    prediction = prediction.bool()
    target = target.bool()

    tp = torch.logical_and(prediction, target).sum().item()
    tn = torch.logical_and(~prediction, ~target).sum().item()
    fp = torch.logical_and(prediction, ~target).sum().item()
    fn = torch.logical_and(~prediction, target).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_binary_metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_split(name, views, gaussians, pipeline, background, dataset, separate_sh, temperature):
    per_view = {}
    aggregate = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    evaluated_views = 0

    for idx, view in enumerate(tqdm(views, desc=f"Semantic metrics ({name})")):
        if view.semantic_map is None:
            continue

        render_pkg = render(
            view,
            gaussians,
            pipeline,
            background,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=separate_sh,
        )
        semantic_image = crop_for_train_test_exp(render_pkg["semantic"], dataset.train_test_exp)
        semantic_gt = crop_for_train_test_exp(view.semantic_map, dataset.train_test_exp)

        prediction, foreground_scores = predict_binary_semantics(
            semantic_image,
            gaussians.get_semantic_prototypes.detach(),
            temperature,
        )
        target = (semantic_gt.squeeze(0) >= 0.5).long()

        if view.alpha_mask is not None:
            alpha_mask = crop_for_train_test_exp(view.alpha_mask, dataset.train_test_exp).squeeze(0) > 0
            prediction = prediction[alpha_mask]
            target = target[alpha_mask]
            foreground_scores = foreground_scores[alpha_mask]
        else:
            foreground_scores = foreground_scores.reshape(-1)
            prediction = prediction.reshape(-1)
            target = target.reshape(-1)

        metrics = compute_binary_metrics(prediction, target)
        metrics["mean_fg_score"] = foreground_scores.float().mean().item() if foreground_scores.numel() > 0 else 0.0
        per_view[f"{idx:05d}_{view.image_name}"] = metrics

        aggregate["tp"] += metrics["tp"]
        aggregate["tn"] += metrics["tn"]
        aggregate["fp"] += metrics["fp"]
        aggregate["fn"] += metrics["fn"]
        evaluated_views += 1

    summary = compute_binary_metrics_from_counts(
        aggregate["tp"],
        aggregate["tn"],
        aggregate["fp"],
        aggregate["fn"],
    ) if evaluated_views > 0 else {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    summary["num_views"] = evaluated_views
    return summary, per_view


def evaluate_semantics(dataset, iteration, pipeline, skip_train, skip_test, separate_sh, temperature, wandb_run=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, semantic_dim=dataset.semantic_dim)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        results = {"summary": {}, "per_view": {}}

        if not skip_train:
            train_summary, train_per_view = evaluate_split(
                "train",
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                dataset,
                separate_sh,
                temperature,
            )
            results["summary"]["train"] = train_summary
            results["per_view"]["train"] = train_per_view

        if not skip_test:
            test_summary, test_per_view = evaluate_split(
                "test",
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                dataset,
                separate_sh,
                temperature,
            )
            results["summary"]["test"] = test_summary
            results["per_view"]["test"] = test_per_view

        results_path = os.path.join(dataset.model_path, "semantic_results.json")
        per_view_path = os.path.join(dataset.model_path, "semantic_per_view.json")

        with open(results_path, "w") as f:
            json.dump(results["summary"], f, indent=2)
        with open(per_view_path, "w") as f:
            json.dump(results["per_view"], f, indent=2)

        print("Semantic evaluation summary:")
        for split_name, split_metrics in results["summary"].items():
            print(
                f"  {split_name}: "
                f"Acc={split_metrics['accuracy']:.4f} "
                f"Prec={split_metrics['precision']:.4f} "
                f"Rec={split_metrics['recall']:.4f} "
                f"F1={split_metrics['f1']:.4f} "
                f"IoU={split_metrics['iou']:.4f} "
                f"Views={split_metrics['num_views']}"
            )
            if wandb_run:
                wandb_run.log({
                    f"semantic_metrics/{split_name}/accuracy": split_metrics["accuracy"],
                    f"semantic_metrics/{split_name}/precision": split_metrics["precision"],
                    f"semantic_metrics/{split_name}/recall": split_metrics["recall"],
                    f"semantic_metrics/{split_name}/f1": split_metrics["f1"],
                    f"semantic_metrics/{split_name}/iou": split_metrics["iou"],
                    f"semantic_metrics/{split_name}/num_views": split_metrics["num_views"],
                })
                per_view_items = list(results["per_view"][split_name].items())[:3]
                for view_name, view_metrics in per_view_items:
                    wandb_run.log({
                        f"semantic_metrics/{split_name}/{view_name}/iou": view_metrics["iou"],
                        f"semantic_metrics/{split_name}/{view_name}/f1": view_metrics["f1"],
                        f"semantic_metrics/{split_name}/{view_name}/accuracy": view_metrics["accuracy"],
                        f"semantic_metrics/{split_name}/{view_name}/mean_fg_score": view_metrics["mean_fg_score"],
                    })
        print(f"Saved semantic metrics to {results_path} and {per_view_path}")


def init_wandb(args):
    if not args.use_wandb:
        return None
    if not WANDB_FOUND:
        print("wandb requested but not installed: not logging to wandb")
        return None
    wandb_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name if args.wandb_run_name else f"semantic-metrics-{os.path.basename(args.model_path)}",
        "entity": args.wandb_entity if args.wandb_entity else None,
        "config": vars(args),
        "dir": args.model_path,
        "job_type": "semantic_metrics",
        "reinit": True,
    }
    return wandb.init(**{k: v for k, v in wandb_kwargs.items() if v is not None})


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Semantic evaluation script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--semantic_temperature", default=0.1, type=float)
    args = get_combined_args(parser)

    print("Evaluating semantics for " + args.model_path)
    safe_state(args.quiet)
    wandb_run = init_wandb(args)

    evaluate_semantics(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        False,
        args.semantic_temperature,
        wandb_run,
    )
    if wandb_run:
        wandb_run.finish()
