import math
import sys
from pathlib import Path

import torch


def add_repo_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "submodules" / "diff-gaussian-rasterization"))
    return repo_root


def build_test_inputs(device: torch.device):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from utils.graphics_utils import getProjectionMatrix

    height = 32
    width = 32
    fovx = math.radians(60.0)
    fovy = math.radians(50.0)

    world_view = torch.eye(4, device=device, dtype=torch.float32)
    projection = getProjectionMatrix(
        znear=0.1,
        zfar=10.0,
        fovX=fovx,
        fovY=fovy,
    ).transpose(0, 1).to(device)
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)

    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=math.tan(fovx * 0.5),
        tanfovy=math.tan(fovy * 0.5),
        bg=torch.tensor([0.1, 0.2, 0.3], device=device),
        scale_modifier=1.0,
        viewmatrix=world_view,
        projmatrix=full_proj,
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    point_count = 8
    means3d = torch.tensor(
        [
            [-0.25, -0.20, 2.0],
            [0.10, -0.10, 2.2],
            [0.00, 0.15, 2.5],
            [0.20, 0.10, 3.0],
            [-0.15, 0.05, 1.8],
            [0.25, -0.05, 2.8],
            [-0.05, 0.25, 3.2],
            [0.05, 0.00, 1.6],
        ],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    means2d = torch.zeros_like(means3d, requires_grad=True)
    colors = torch.rand(point_count, 3, device=device, dtype=torch.float32, requires_grad=True)
    semantics = torch.linspace(0.1, 0.8, point_count, device=device, dtype=torch.float32).unsqueeze(1).requires_grad_()
    opacities = torch.full((point_count, 1), 0.7, device=device, dtype=torch.float32, requires_grad=True)
    scales = torch.full((point_count, 3), 0.08, device=device, dtype=torch.float32, requires_grad=True)
    rotations = torch.zeros(point_count, 4, device=device, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        rotations[:, 0] = 1.0

    return rasterizer, means3d, means2d, colors, semantics, opacities, scales, rotations


def main():
    repo_root = add_repo_paths()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this script on a Linux machine with an NVIDIA GPU.")

    try:
        from diff_gaussian_rasterization import GaussianRasterizer  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise SystemExit(
            "Failed to import diff_gaussian_rasterization. "
            "Build it first, for example:\n"
            f"  cd {repo_root / 'submodules' / 'diff-gaussian-rasterization'}\n"
            "  python setup.py build_ext --inplace\n"
            f"Original error: {exc}"
        ) from exc

    device = torch.device("cuda")
    torch.manual_seed(0)

    rasterizer, means3d, means2d, colors, semantics, opacities, scales, rotations = build_test_inputs(device)

    image, radii, depth = rasterizer(
        means3D=means3d,
        means2D=means2d,
        shs=None,
        colors_precomp=colors,
        semantics=semantics,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    assert image.shape == (4, 32, 32), f"Unexpected image shape: {tuple(image.shape)}"
    assert depth.shape == (1, 32, 32), f"Unexpected depth shape: {tuple(depth.shape)}"
    assert radii.shape == (8,), f"Unexpected radii shape: {tuple(radii.shape)}"
    assert torch.isfinite(image).all(), "Rendered image contains non-finite values."
    assert torch.isfinite(depth).all(), "Depth output contains non-finite values."
    assert (radii > 0).any(), "No Gaussian contributed to the frame."

    rgb = image[:3]
    semantic = image[3:4]
    loss = rgb.mean() + semantic.mean() + 0.1 * depth.mean()
    loss.backward()

    tensors_to_check = {
        "means3d": means3d.grad,
        "means2d": means2d.grad,
        "colors": colors.grad,
        "semantics": semantics.grad,
        "opacities": opacities.grad,
        "scales": scales.grad,
        "rotations": rotations.grad,
    }

    for name, grad in tensors_to_check.items():
        assert grad is not None, f"{name} grad is missing."
        assert torch.isfinite(grad).all(), f"{name} grad contains non-finite values."

    assert semantics.grad.abs().sum().item() > 0, "Semantic gradients are all zero."

    print("Semantic rasterizer validation passed.")
    print(f"render shape: {tuple(image.shape)}")
    print(f"depth shape: {tuple(depth.shape)}")
    print(f"semantic grad sum: {semantics.grad.abs().sum().item():.6f}")
    print(f"visible gaussians: {(radii > 0).sum().item()} / {radii.numel()}")


if __name__ == "__main__":
    main()
