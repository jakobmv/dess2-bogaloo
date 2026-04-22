from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from dess2_bogaloo.dess_updated import get_mu_sigma


def _make_three_layer_mlp(
    input_dim: int,
    *,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass(frozen=True)
class DESSOutputs:
    mu: torch.Tensor
    sigma: torch.Tensor
    pred: torch.Tensor
    mu_for_sigma: torch.Tensor


class TextDessAdapter(nn.Module):
    """Variant 1: one 3-layer MLP jointly predicts mu and sigma."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        target_dim = output_dim or input_dim
        self.network = _make_three_layer_mlp(
            input_dim,
            hidden_dim=hidden_dim,
            output_dim=target_dim * 2,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> DESSOutputs:
        pred = self.network(x)
        mu, sigma = get_mu_sigma(pred)
        return DESSOutputs(
            mu=mu,
            sigma=sigma,
            pred=pred,
            mu_for_sigma=mu,
        )


class FrozenMuSigmaAdapter(nn.Module):
    """Variant 2: mu is the frozen encoder output, only sigma is learned."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        target_dim = output_dim or input_dim
        self.sigma_head = _make_three_layer_mlp(
            input_dim,
            hidden_dim=hidden_dim,
            output_dim=target_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> DESSOutputs:
        mu = x
        sigma_pred = self.sigma_head(x)
        pred = torch.cat((mu, sigma_pred), dim=-1)
        _, sigma = get_mu_sigma(pred)
        return DESSOutputs(
            mu=mu,
            sigma=sigma,
            pred=pred,
            mu_for_sigma=mu.detach(),
        )


class DualHeadDessAdapter(nn.Module):
    """Variant 3: separate mu and sigma heads, sigma depends on detached mu."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        target_dim = output_dim or input_dim
        self.mu_head = _make_three_layer_mlp(
            input_dim,
            hidden_dim=hidden_dim,
            output_dim=target_dim,
            dropout=dropout,
        )
        self.sigma_head = _make_three_layer_mlp(
            input_dim,
            hidden_dim=hidden_dim,
            output_dim=target_dim,
            dropout=dropout,
        )

    def sigma_input(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> DESSOutputs:
        mu = self.mu_head(x)
        mu_for_sigma = mu.detach()
        sigma_pred = self.sigma_head(self.sigma_input(x, mu_for_sigma))
        pred = torch.cat((mu, sigma_pred), dim=-1)
        _, sigma = get_mu_sigma(pred)
        return DESSOutputs(
            mu=mu,
            sigma=sigma,
            pred=pred,
            mu_for_sigma=mu_for_sigma,
        )


class DualHeadQueryConcatSigmaAdapter(DualHeadDessAdapter):
    """Variant 4: sigma head sees both the query embedding and detached mu."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        output_dim: int | None = None,
    ) -> None:
        target_dim = output_dim or input_dim
        nn.Module.__init__(self)
        self.mu_head = _make_three_layer_mlp(
            input_dim,
            hidden_dim=hidden_dim,
            output_dim=target_dim,
            dropout=dropout,
        )
        self.sigma_head = _make_three_layer_mlp(
            input_dim + target_dim,
            hidden_dim=hidden_dim,
            output_dim=target_dim,
            dropout=dropout,
        )

    def sigma_input(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, mu), dim=-1)


VARIANT_MODEL_TYPES = {
    "mlp_joint": TextDessAdapter,
    "frozen_mu_sigma_mlp": FrozenMuSigmaAdapter,
    "dual_head_detached_sigma": DualHeadDessAdapter,
    "dual_head_query_concat_sigma": DualHeadQueryConcatSigmaAdapter,
}


class DESSHead(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return get_mu_sigma(x)


class DessReranker(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        output_dim: int | None = None,
        variant: str = "mlp_joint",
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        model_type = VARIANT_MODEL_TYPES[variant]
        self.backbone = model_type(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(x)
        return outputs.mu, outputs.sigma


ClipTextDessAdapter = TextDessAdapter
