# NOTE: This file vendors the official DESS implementation from
# https://github.com/neddi-as/DESS-Dimensional-PDFs-for-Embedding-Space-Sampling
# and must not be modified locally. If integration fixes are needed in this repo,
# add them in `src/dess2_bogaloo/dess_updated.py` instead of editing this file.

import torch
import torch.nn as nn
from torch.nn import _reduction as _Reduction
import torch.nn.functional as F

# DESS implemented in the style of loss functions found in the PyTorch libary. In place of F.loss use the provided 'F_dess_loss' function.

# Copy of base Loss class in PyTorch
class _Loss(nn.Module):
    reduction: str  # Choose to return loss as "mean", "sum" or "none". none means element-wise loss.

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


def get_mu_sigma(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Split tensor in equal parts.
    mu, sigma = pred.chunk(2, dim=1)

    return mu, sigma


def single_target_criterion(
    mu_preds: torch.Tensor,
    sigma_preds: torch.Tensor,
    targets: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Element-wise mean squared error between target components and mu predictions
    # mu_losses = F.mse_loss(targets, mu_preds, reduction="none")

    mu_losses = F.mse_loss(mu_preds, targets, reduction="none")
    # Element-wise absolute error between target components and mu predictions
    abs_errors = F.l1_loss(targets, mu_preds, reduction="none")

    # Element-wise mean squared error between beta-scaled absolute errors and sigma predictions
    sigma_losses = F.l1_loss(abs_errors * beta, sigma_preds, reduction="none")

    return mu_losses, sigma_losses


def multi_target_criterion(
    mu_pred: torch.Tensor,
    sigma_pred: torch.Tensor,
    target: torch.Tensor,
    beta: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_means = torch.mean(target, dim=1)

    mu_losses = F.mse_loss(mu_pred, y_means, reduction="none")

    delta_Y_i = torch.max(target, dim=1)[0] - torch.min(target, dim=1)[0]
    abs_errors = F.l1_loss(y_means, mu_pred, reduction="none")

    sigma_target = alpha * (abs_errors * beta) + (1 - alpha) * delta_Y_i * beta
    sigma_losses = F.mse_loss(sigma_pred, sigma_target, reduction="none")

    return mu_losses, sigma_losses


class DESSLoss(_Loss):
    """DESSLoss, a loss function with two parts. mu_loss, a reconstruction loss, ie. MSE, and
     sigma_loss, which attepmts to guess at the error rate or the reconstruction loss to learn an
     uncertainty estimate. In essence, it lears a Gaussian distribution for each input, so that
     predictions from similar inputs will be sampled from similar multi-dimensional Gaussian distributions.

    Args:
        pred (torch.Tensor): Shape (batch_dim, 2*emb_dim).
        y (torch.Tensor): Shape (batch_dim, emb_dim, num_targets).
        beta (float): Weight for the uncertainty term. Default: 1.0
        alpha (float): Weight for balancing between mu and sigma loss. Default: 0.5
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
    """

    def __init__(
        self,
        beta: float = 1.0,
        alpha: float = 0.5,
        mu_loss: str = "mse",
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.beta = beta
        self.alpha = alpha
        self.mu_loss = mu_loss

    def forward(self, pred: torch.Tensor, y: torch.Tensor):
        return F_dess_loss(
            pred,
            y,
            beta=self.beta,
            alpha=self.alpha,
            reduction=self.reduction,
        )


# F_ to mimic F.loss from pytorch
def F_dess_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    alpha: float = 0.5,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the DESS loss.

    Args:
        pred: Model predictions of shape (batch_dim, 2*emb_dim)
        target: Ground truth targets of shape (batch_dim, emb_dim) or (batch_dim, emb_dim, num_targets)
        beta: Weight for uncertainty term
        alpha: Weight for balancing terms in multi-target-sigma-loss
        reduction: Reduction method for the loss ('none', 'mean', or 'sum')
        mu_loss: Type of loss function for mean predictions. Defaults to "mse"

    Returns:
        A tuple of (total_loss, mu_loss, sigma_loss) where:
            - total_loss: Combined loss (mu_loss + alpha * sigma_loss)
            - mu_loss: Mean prediction loss component
            - sigma_loss: Uncertainty loss component
    """

    mu_pred, sigma_pred = get_mu_sigma(pred)

    if len(target.shape) == 2:  # If shape is (batch_dim, emb_dim)
        mu_loss, sigma_loss = single_target_criterion(mu_pred, sigma_pred, target, beta)
    elif len(target.shape) == 3:  # If shape is (batch_dim, emb_dim, num_targets)
        mu_loss, sigma_loss = multi_target_criterion(
            mu_pred, sigma_pred, target, beta, alpha, mu_loss
        )
    else:
        raise ValueError(
            f"Target shape {target.shape} not supported. "
            "Expected shape (batch_dim, emb_dim) for single target or "
            "(batch_dim, emb_dim, num_targets) for multiple targets"
        )

    combined_loss = torch.cat((mu_loss, sigma_loss), dim=-1)

    if reduction == "none":
        return combined_loss, mu_loss, sigma_loss
    elif reduction == "mean":
        return combined_loss.mean(), mu_loss.mean(), sigma_loss.mean()
    else:  # sum
        return combined_loss.sum(), mu_loss.sum(), sigma_loss.sum()


class DESSLayer(nn.Module):
    def forward(self, x):
        mu, sigma = x.chunk(2, dim=-1)
        sigma = F.softplus(sigma)
        return torch.cat((mu, sigma), dim=-1)


class DESSModel(nn.Module):
    def __init__(self, input_dim: int = 3, embedding_dim: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim * 2),
            DESSLayer(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.network(x)
