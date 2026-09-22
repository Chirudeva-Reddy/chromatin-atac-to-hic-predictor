r"""Edge-aware structural loss module for 3D chromatin conformation prediction.

References:
    - Zhang, S., Chadwick, R. Y., & Felix, M. A. (2019). HiC-Reg: predicting
      high-resolution 3D chromatin conformation from 1D genomic and epigenomic
      features. Nature Communications, 10, 3995.
    - Yang, T., Zhang, F., Yardimci, G. G., Song, F., Hardison, R. C.,
      Noble, W. S., Yue, F., & Li, Q. (2017). HiCRep: assessing the reproducibility
      of Hi-C data using a stratum-adjusted correlation coefficient.
      Bioinformatics, 33(14), 2196-2204.
    - Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-Image
      Translation with Conditional Adversarial Networks. CVPR 2017.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralChromatinLoss(nn.Module):
    r"""Baseline structural chromatin loss with diagonal weighting."""

    def __init__(self, diagonal_weight: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.diagonal_weight = diagonal_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_mse = self.mse(pred, target)
        bins = pred.shape[-1]
        weight_matrix = torch.eye(bins, device=pred.device) * 3.0 + 1.0
        weighted_loss = torch.mean(weight_matrix * (pred - target) ** 2)
        return base_mse + self.diagonal_weight * weighted_loss


class EdgeAwareChromatinLoss(nn.Module):
    r"""Edge-aware structural loss incorporating 2D spatial filtering.

    Combines standard reconstruction loss with high-frequency edge detection
    via a 2D Laplacian or Sobel filter to enforce sharp Topologically Associating
    Domain (TAD) boundaries and off-diagonal chromatin loops.

    Mathematical Formulation:
    -------------------------
    2D discrete Laplacian filter:
        K = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    Edge feature maps:
        \Delta \hat{Y} = \text{Conv2D}(\hat{Y}, K)
        \Delta Y = \text{Conv2D}(Y, K)
    Edge loss:
        \mathcal{L}_{\text{edge}} = \| \Delta \hat{Y} - \Delta Y \|_1
    Total loss:
        \mathcal{L} = \mathcal{L}_{\text{base}} + \alpha \cdot \mathcal{L}_{\text{edge}}

    Args:
        edge_weight: Weight \alpha for the edge detection loss component. Default: 0.5.
        filter_type: 'laplacian' (default) or 'sobel'.
        diagonal_weight: Weighting factor for near-diagonal contacts. Default: 0.5.
        loss_norm: 'l1' (default) or 'l2' for edge residuals.
    """

    def __init__(
        self,
        edge_weight: float = 0.5,
        filter_type: str = "laplacian",
        diagonal_weight: float = 0.5,
        loss_norm: str = "l1",
    ):
        super().__init__()
        if edge_weight < 0.0:
            raise ValueError(f"edge_weight must be non-negative, got {edge_weight}")
        if filter_type not in ("laplacian", "sobel"):
            raise ValueError(f"filter_type must be 'laplacian' or 'sobel', got {filter_type}")
        if loss_norm not in ("l1", "l2"):
            raise ValueError(f"loss_norm must be 'l1' or 'l2', got {loss_norm}")

        self.edge_weight = float(edge_weight)
        self.filter_type = filter_type
        self.diagonal_weight = float(diagonal_weight)
        self.loss_norm = loss_norm
        self.mse = nn.MSELoss()

        if filter_type == "laplacian":
            # 3x3 discrete Laplacian kernel
            laplacian_kernel = torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, -4.0, 1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ).view(1, 1, 3, 3)
            self.register_buffer("kernel", laplacian_kernel)
        else:
            # Sobel horizontal and vertical kernels
            sobel_x = torch.tensor(
                [
                    [-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ).view(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [
                    [-1.0, -2.0, -1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 1.0],
                ],
                dtype=torch.float32,
            ).view(1, 1, 3, 3)
            self.register_buffer("kernel_x", sobel_x)
            self.register_buffer("kernel_y", sobel_y)

    def _extract_edges(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure (B, 1, H, W) shape
        if x.dim() == 2:
            x_4d = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x_4d = x.unsqueeze(1)
        elif x.dim() == 4 and x.size(1) == 1:
            x_4d = x
        else:
            raise ValueError(f"Expected tensor of dim 2 (H, W), 3 (B, H, W) or 4 (B, 1, H, W), got {x.shape}")

        if self.filter_type == "laplacian":
            return F.conv2d(x_4d, self.kernel, padding=1)
        else:
            gx = F.conv2d(x_4d, self.kernel_x, padding=1)
            gy = F.conv2d(x_4d, self.kernel_y, padding=1)
            return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        r"""Compute edge-aware chromatin loss.

        Args:
            pred: Predicted contact map of shape ``(B, H, W)`` or ``(B, 1, H, W)``.
            target: Ground truth contact map of shape ``(B, H, W)`` or ``(B, 1, H, W)``.
            return_components: If True, returns ``(total_loss, components_dict)``.

        Returns:
            Scalar loss tensor (or tuple with components dict).
        """
        # 1. Base reconstruction loss
        base_mse = self.mse(pred, target)
        if self.diagonal_weight > 0.0:
            bins = pred.shape[-1]
            weight_matrix = torch.eye(bins, device=pred.device) * 3.0 + 1.0
            weighted_diag = torch.mean(weight_matrix * (pred - target) ** 2)
            base_loss = base_mse + self.diagonal_weight * weighted_diag
        else:
            base_loss = base_mse

        # 2. Edge detection loss
        if self.edge_weight > 0.0:
            pred_edges = self._extract_edges(pred)
            target_edges = self._extract_edges(target)

            if self.loss_norm == "l1":
                edge_loss = F.l1_loss(pred_edges, target_edges)
            else:
                edge_loss = F.mse_loss(pred_edges, target_edges)
        else:
            edge_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        total_loss = base_loss + self.edge_weight * edge_loss

        if return_components:
            components = {
                "base_loss": base_loss.detach(),
                "edge_loss": edge_loss.detach(),
                "total_loss": total_loss.detach(),
            }
            return total_loss, components

        return total_loss


def compute_stratum_correlation(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    max_diag: int = 30,
) -> float:
    r"""Compute the Stratum-Adjusted Correlation Coefficient (SCC) proxy.

    Evaluates the Pearson correlation stratified across diagonal interaction offsets
    to eliminate genomic distance decay bias, as described in HiCRep (Yang et al., 2017).

    Args:
        y_true: Ground truth 2D contact matrix.
        y_pred: Predicted 2D contact matrix.
        max_diag: Maximum diagonal offset to evaluate.

    Returns:
        Mean stratum-adjusted correlation coefficient across valid strata.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    if y_true.ndim == 4 and y_true.shape[1] == 1:
        y_true = y_true.squeeze(1)
    if y_pred.ndim == 4 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(1)

    if y_true.ndim == 3:
        # Batch of matrices: compute mean across batch
        return float(np.mean([
            compute_stratum_correlation(y_true[i], y_pred[i], max_diag=max_diag)
            for i in range(y_true.shape[0])
        ]))

    strata_rs = []
    for k in range(1, max_diag + 1):
        diag_true = np.diag(y_true, k=k)
        diag_pred = np.diag(y_pred, k=k)
        if np.std(diag_true) > 1e-6 and np.std(diag_pred) > 1e-6:
            r_k = np.corrcoef(diag_true, diag_pred)[0, 1]
            if not np.isnan(r_k):
                strata_rs.append(r_k)

    return float(np.mean(strata_rs)) if strata_rs else 0.0
