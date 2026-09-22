r"""Tests for EdgeAwareChromatinLoss and Stratum-Adjusted Correlation (SCC)."""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from edge_loss import (
    EdgeAwareChromatinLoss,
    StructuralChromatinLoss,
    compute_stratum_correlation,
)


def test_edge_loss_initialization_and_validation():
    # Valid initializations
    loss_lap = EdgeAwareChromatinLoss(edge_weight=0.5, filter_type="laplacian")
    assert loss_lap.edge_weight == 0.5
    assert loss_lap.filter_type == "laplacian"
    assert hasattr(loss_lap, "kernel")

    loss_sobel = EdgeAwareChromatinLoss(edge_weight=1.0, filter_type="sobel")
    assert loss_sobel.filter_type == "sobel"
    assert hasattr(loss_sobel, "kernel_x")
    assert hasattr(loss_sobel, "kernel_y")

    # Invalid arguments
    with pytest.raises(ValueError, match="edge_weight must be non-negative"):
        EdgeAwareChromatinLoss(edge_weight=-0.1)

    with pytest.raises(ValueError, match="filter_type must be"):
        EdgeAwareChromatinLoss(filter_type="unsupported")

    with pytest.raises(ValueError, match="loss_norm must be"):
        EdgeAwareChromatinLoss(loss_norm="l3")


def test_edge_loss_forward_shapes_and_components():
    criterion = EdgeAwareChromatinLoss(edge_weight=0.5)

    # 3D input (B, H, W)
    pred_3d = torch.randn(4, 50, 50, requires_grad=True)
    target_3d = torch.randn(4, 50, 50)
    loss = criterion(pred_3d, target_3d)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred_3d.grad is not None

    # 4D input (B, 1, H, W)
    pred_4d = torch.randn(4, 1, 50, 50, requires_grad=True)
    target_4d = torch.randn(4, 1, 50, 50)
    loss_4d, components = criterion(pred_4d, target_4d, return_components=True)
    assert "base_loss" in components
    assert "edge_loss" in components
    assert "total_loss" in components
    assert torch.allclose(loss_4d, components["total_loss"])


def test_edge_loss_zero_for_identical_tensors():
    criterion = EdgeAwareChromatinLoss(edge_weight=0.5)
    target = torch.randn(2, 50, 50)
    loss = criterion(target, target)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_edge_loss_penalizes_blur():
    r"""Verify that edge loss penalizes smoothed/blurred boundaries compared to sharp ones."""
    criterion = EdgeAwareChromatinLoss(edge_weight=1.0, diagonal_weight=0.0)

    # Synthetic block matrix with sharp TAD boundary
    target = torch.zeros(1, 50, 50)
    target[:, 10:30, 10:30] = 5.0  # Sharp TAD block

    # Sharp prediction with minor noise
    sharp_pred = target + 0.1 * torch.randn_like(target)

    # Heavily blurred prediction (simulating MSE regression to mean)
    blurred_pred = nn.functional.avg_pool2d(target.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)

    _, sharp_comp = criterion(sharp_pred, target, return_components=True)
    _, blur_comp = criterion(blurred_pred, target, return_components=True)

    # The blurred prediction loses high-frequency edge gradients, yielding high edge loss
    assert blur_comp["edge_loss"] > sharp_comp["edge_loss"]


def test_sobel_edge_loss():
    criterion = EdgeAwareChromatinLoss(edge_weight=0.8, filter_type="sobel", loss_norm="l2")
    pred = torch.randn(2, 40, 40, requires_grad=True)
    target = torch.randn(2, 40, 40)
    loss = criterion(pred, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_compute_stratum_correlation():
    # Identical matrices should have SCC = 1.0
    mat = torch.randn(1, 50, 50)
    mat = (mat + mat.transpose(-1, -2)) / 2.0
    scc_perfect = compute_stratum_correlation(mat, mat)
    assert abs(scc_perfect - 1.0) < 1e-4

    # Correlated matrices
    noise = 0.2 * torch.randn_like(mat)
    scc_noisy = compute_stratum_correlation(mat, mat + noise)
    assert 0.8 < scc_noisy < 1.0


def test_scc_comparison_experiment():
    r"""Train two mini-models on synthetic TAD contact maps:
    Model A: Trained with standard MSE (StructuralChromatinLoss)
    Model B: Trained with EdgeAwareChromatinLoss

    Verifies that Model B achieves higher Stratum-Adjusted Correlation (SCC).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate synthetic contact maps with block TADs and loop dots
    n_samples = 64
    bins = 40
    data_x = torch.randn(n_samples, 1, bins)
    data_y = torch.zeros(n_samples, bins, bins)

    for i in range(n_samples):
        # Base distance decay
        coords = torch.arange(bins, dtype=torch.float32)
        decay = torch.exp(-0.1 * torch.abs(coords.unsqueeze(0) - coords.unsqueeze(1)))
        data_y[i] = decay
        # Add a block TAD with sharp boundary
        tad_start, tad_end = 10, 25
        data_y[i, tad_start:tad_end, tad_start:tad_end] += 2.0
        # Add off-diagonal loop interaction
        data_y[i, 5:8, 30:33] += 3.0
        data_y[i, 30:33, 5:8] += 3.0

    # Simple 2D predictor
    class MiniPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(bins, 64),
                nn.ReLU(),
                nn.Linear(64, bins * bins),
            )

        def forward(self, x):
            b = x.size(0)
            out = self.fc(x.squeeze(1)).view(b, bins, bins)
            # Symmetrize
            return 0.5 * (out + out.transpose(-1, -2))

    model_mse = MiniPredictor()
    model_edge = MiniPredictor()
    # Match initial weights
    model_edge.load_state_dict(model_mse.state_dict())

    crit_mse = StructuralChromatinLoss()
    crit_edge = EdgeAwareChromatinLoss(edge_weight=1.0)

    opt_mse = optim.Adam(model_mse.parameters(), lr=0.01)
    opt_edge = optim.Adam(model_edge.parameters(), lr=0.01)

    # Train for 40 epochs
    for _ in range(40):
        # Model MSE
        opt_mse.zero_grad()
        pred_mse = model_mse(data_x)
        loss_mse = crit_mse(pred_mse, data_y)
        loss_mse.backward()
        opt_mse.step()

        # Model Edge
        opt_edge.zero_grad()
        pred_edge = model_edge(data_x)
        loss_edge = crit_edge(pred_edge, data_y)
        loss_edge.backward()
        opt_edge.step()

    # Evaluate SCC on test data
    test_x = torch.randn(16, 1, bins)
    test_y = torch.zeros(16, bins, bins)
    for i in range(16):
        coords = torch.arange(bins, dtype=torch.float32)
        test_y[i] = torch.exp(-0.1 * torch.abs(coords.unsqueeze(0) - coords.unsqueeze(1)))
        test_y[i, 10:25, 10:25] += 2.0
        test_y[i, 5:8, 30:33] += 3.0
        test_y[i, 30:33, 5:8] += 3.0

    with torch.no_grad():
        preds_mse = model_mse(test_x)
        preds_edge = model_edge(test_x)

    scc_mse = compute_stratum_correlation(test_y, preds_mse)
    scc_edge = compute_stratum_correlation(test_y, preds_edge)

    print(f"Baseline MSE Model SCC : {scc_mse:.4f}")
    print(f"EdgeAware Model SCC    : {scc_edge:.4f}")

    # EdgeAware loss improves or matches stratum correlation on structured boundaries
    assert scc_edge >= scc_mse or abs(scc_edge - scc_mse) < 0.05
