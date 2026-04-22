from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mu_sigma(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu, raw_sigma = pred.chunk(2, dim=-1)
    sigma = F.softplus(raw_sigma) + 1e-6
    return mu, sigma


def single_target_criterion(
    mu_pred: torch.Tensor,
    sigma_pred: torch.Tensor,
    target: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mu_loss = F.mse_loss(mu_pred, target, reduction="none")
    abs_error = F.l1_loss(mu_pred, target, reduction="none")
    sigma_target = abs_error * beta
    sigma_loss = F.l1_loss(sigma_pred, sigma_target, reduction="none")
    return mu_loss, sigma_loss


def multi_target_criterion(
    mu_pred: torch.Tensor,
    sigma_pred: torch.Tensor,
    target: torch.Tensor,
    beta: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.ndim != 3:
        raise ValueError(f"Expected 3D target tensor, got shape {tuple(target.shape)}")

    target_mean = target.mean(dim=1)
    target_span = target.max(dim=1).values - target.min(dim=1).values
    mu_loss = F.mse_loss(mu_pred, target_mean, reduction="none")
    abs_error = F.l1_loss(mu_pred, target_mean, reduction="none")
    sigma_target = alpha * (abs_error * beta) + (1.0 - alpha) * (target_span * beta)
    sigma_loss = F.mse_loss(sigma_pred, sigma_target, reduction="none")
    return mu_loss, sigma_loss


def single_target_sigma_loss(
    sigma_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mu_for_sigma: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    abs_error = F.l1_loss(mu_for_sigma, target, reduction="none")
    sigma_target = abs_error * beta
    return F.l1_loss(sigma_pred, sigma_target, reduction="none")


def multi_target_sigma_loss(
    sigma_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mu_for_sigma: torch.Tensor,
    beta: float,
    alpha: float,
) -> torch.Tensor:
    if target.ndim != 3:
        raise ValueError(f"Expected 3D target tensor, got shape {tuple(target.shape)}")
    target_mean = target.mean(dim=1)
    target_span = target.max(dim=1).values - target.min(dim=1).values
    abs_error = F.l1_loss(mu_for_sigma, target_mean, reduction="none")
    sigma_target = alpha * (abs_error * beta) + (1.0 - alpha) * (target_span * beta)
    return F.mse_loss(sigma_pred, sigma_target, reduction="none")


def dess_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 0.5,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu_pred, sigma_pred = get_mu_sigma(pred)

    if target.ndim == 2:
        mu_loss, sigma_loss = single_target_criterion(mu_pred, sigma_pred, target, beta)
    elif target.ndim == 3:
        mu_loss, sigma_loss = multi_target_criterion(mu_pred, sigma_pred, target, beta, alpha)
    else:
        raise ValueError(
            f"Target shape {tuple(target.shape)} not supported. Expected 2D or 3D target."
        )

    combined = torch.cat((mu_loss, sigma_loss), dim=-1)
    if reduction == "none":
        return combined, mu_loss, sigma_loss
    if reduction == "sum":
        return combined.sum(), mu_loss.sum(), sigma_loss.sum()
    if reduction != "mean":
        raise ValueError(f"Unsupported reduction: {reduction}")
    return combined.mean(), mu_loss.mean(), sigma_loss.mean()


def dess_loss_from_parts(
    mu_pred: torch.Tensor,
    sigma_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 0.5,
    reduction: str = "mean",
    mu_for_sigma: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_reference = mu_pred if mu_for_sigma is None else mu_for_sigma

    if target.ndim == 2:
        mu_loss = F.mse_loss(mu_pred, target, reduction="none")
        sigma_loss = single_target_sigma_loss(
            sigma_pred,
            target,
            mu_for_sigma=sigma_reference,
            beta=beta,
        )
    elif target.ndim == 3:
        mu_target = target.mean(dim=1)
        mu_loss = F.mse_loss(mu_pred, mu_target, reduction="none")
        sigma_loss = multi_target_sigma_loss(
            sigma_pred,
            target,
            mu_for_sigma=sigma_reference,
            beta=beta,
            alpha=alpha,
        )
    else:
        raise ValueError(
            f"Target shape {tuple(target.shape)} not supported. Expected 2D or 3D target."
        )

    combined = torch.cat((mu_loss, sigma_loss), dim=-1)
    if reduction == "none":
        return combined, mu_loss, sigma_loss
    if reduction == "sum":
        return combined.sum(), mu_loss.sum(), sigma_loss.sum()
    if reduction != "mean":
        raise ValueError(f"Unsupported reduction: {reduction}")
    return combined.mean(), mu_loss.mean(), sigma_loss.mean()


class DESSLoss(nn.Module):
    def __init__(
        self,
        *,
        beta: float = 1.0,
        alpha: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return dess_loss(
            pred,
            target,
            beta=self.beta,
            alpha=self.alpha,
            reduction=self.reduction,
        )


def gaussian_log_score(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    variance = sigma.square()
    normalized = (target - mu).square() / variance
    log_det = torch.log(variance)
    return -0.5 * (normalized + log_det).mean(dim=-1)
