from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from dess2_bogaloo.dess_model import (
    DESSOutputs,
    DualHeadDessAdapter,
    DualHeadQueryConcatSigmaAdapter,
    TextDessAdapter,
)
from dess2_bogaloo.dess_updated import dess_loss_from_parts
from dess2_bogaloo.utils import ensure_dir, write_json


GAS_TURBINE_URL = (
    "https://archive.ics.uci.edu/static/public/551/"
    "gas+turbine+co+and+nox+emission+data+set.zip"
)
GAS_TURBINE_CITATION = (
    "Gas Turbine CO and NOx Emission Data Set [Dataset]. (2019). "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5WC95."
)
GAS_TURBINE_YEAR_FILES = [
    "gt_2011.csv",
    "gt_2012.csv",
    "gt_2013.csv",
    "gt_2014.csv",
    "gt_2015.csv",
]
FEATURE_COLUMNS = [
    "AT",
    "AP",
    "AH",
    "AFDP",
    "GTEP",
    "TIT",
    "TAT",
    "TEY",
    "CDP",
]
TARGET_COLUMNS = ["CO", "NOX"]
SINGLE_TARGET_VARIANT_DESCRIPTIONS = {
    "mlp_joint": "One 3-layer MLP jointly predicts the target mean vector and uncertainty vector.",
    "frozen_mu_sigma_mlp": (
        "Uses a frozen linear mean regressor for mu and learns only sigma with a 3-layer MLP. "
        "This adapts the fixed-mu idea to tabular regression where inputs and targets do not share a space."
    ),
    "dual_head_detached_sigma": "Separate mu and sigma heads; sigma is trained against a detached copy of mu.",
    "dual_head_query_concat_sigma": (
        "Separate heads; sigma receives the concatenation of the input features and detached mu."
    ),
}


@dataclass(frozen=True)
class GasTurbinePaths:
    root: Path

    @property
    def dataset_dir(self) -> Path:
        return self.root / "raw" / "gas_turbine_co_nox"

    @property
    def archive_path(self) -> Path:
        return self.dataset_dir / "gas_turbine_co_nox.zip"

    def year_path(self, filename: str) -> Path:
        return self.dataset_dir / filename


@dataclass(frozen=True)
class SingleTargetConfig:
    batch_size: int = 512
    eval_batch_size: int = 4096
    epochs: int = 40
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta: float = 1.0
    alpha: float = 0.5
    dropout: float = 0.1
    hidden_dim: int = 128
    val_fraction: float = 0.2
    seed: int = 42
    device: str = "cpu"
    variant: str = "mlp_joint"


class _ArrayDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.targets = torch.from_numpy(targets.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def download_gas_turbine_data(paths: GasTurbinePaths, *, overwrite: bool = False) -> list[Path]:
    ensure_dir(paths.dataset_dir)
    expected_paths = [paths.year_path(name) for name in GAS_TURBINE_YEAR_FILES]
    if not overwrite and all(path.exists() for path in expected_paths):
        return expected_paths

    payload = urlopen(GAS_TURBINE_URL).read()
    paths.archive_path.write_bytes(payload)
    extracted: list[Path] = []
    with ZipFile(BytesIO(payload)) as archive:
        for name in GAS_TURBINE_YEAR_FILES:
            destination = paths.year_path(name)
            if overwrite or not destination.exists():
                destination.write_bytes(archive.read(name))
            extracted.append(destination)
    return extracted


def load_gas_turbine_frames(paths: GasTurbinePaths) -> dict[str, pd.DataFrame]:
    download_gas_turbine_data(paths)
    frames: dict[str, pd.DataFrame] = {}
    for name in GAS_TURBINE_YEAR_FILES:
        year = name.removeprefix("gt_").removesuffix(".csv")
        frame = pd.read_csv(paths.year_path(name))
        frame["year"] = int(year)
        frames[year] = frame
    return frames


def build_gas_turbine_splits(
    paths: GasTurbinePaths,
    *,
    val_fraction: float = 0.2,
) -> dict[str, pd.DataFrame]:
    frames = load_gas_turbine_frames(paths)
    train_pool = pd.concat(
        [frames["2011"], frames["2012"], frames["2013"]],
        axis=0,
        ignore_index=True,
    )
    test = pd.concat(
        [frames["2014"], frames["2015"]],
        axis=0,
        ignore_index=True,
    )
    val_rows = max(1, int(round(train_pool.shape[0] * val_fraction)))
    train = train_pool.iloc[:-val_rows].reset_index(drop=True)
    val = train_pool.iloc[-val_rows:].reset_index(drop=True)
    return {"train": train, "val": val, "test": test.reset_index(drop=True)}


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


class FrozenLinearMuSigmaAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        regressor = LinearRegression()
        regressor.fit(train_features, train_targets)
        self.mu_head = nn.Linear(input_dim, output_dim)
        with torch.no_grad():
            self.mu_head.weight.copy_(torch.from_numpy(regressor.coef_.astype(np.float32)))
            self.mu_head.bias.copy_(torch.from_numpy(regressor.intercept_.astype(np.float32)))
        for parameter in self.mu_head.parameters():
            parameter.requires_grad = False
        self.sigma_head = _make_three_layer_mlp(
            input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> DESSOutputs:
        mu = self.mu_head(x)
        sigma_pred = self.sigma_head(x)
        sigma = torch.nn.functional.softplus(sigma_pred) + 1e-6
        pred = torch.cat((mu, sigma_pred), dim=-1)
        return DESSOutputs(
            mu=mu,
            sigma=sigma,
            pred=pred,
            mu_for_sigma=mu.detach(),
        )


def _build_model(
    *,
    variant: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> nn.Module:
    if variant == "mlp_joint":
        return TextDessAdapter(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if variant == "frozen_mu_sigma_mlp":
        return FrozenLinearMuSigmaAdapter(
            input_dim=input_dim,
            output_dim=output_dim,
            train_features=train_features,
            train_targets=train_targets,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if variant == "dual_head_detached_sigma":
        return DualHeadDessAdapter(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if variant == "dual_head_query_concat_sigma":
        return DualHeadQueryConcatSigmaAdapter(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported single-target variant: {variant}")


def _jsonable_config(config: SingleTargetConfig) -> dict[str, object]:
    return asdict(config)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _target_metrics(
    *,
    targets_orig: np.ndarray,
    mu_orig: np.ndarray,
    sigma_orig: np.ndarray,
) -> dict[str, float]:
    error = mu_orig - targets_orig
    abs_error = np.abs(error)
    sq_error = np.square(error)

    rmse_per_target = np.sqrt(np.mean(sq_error, axis=0))
    mae_per_target = np.mean(abs_error, axis=0)
    target_mean = np.mean(targets_orig, axis=0, keepdims=True)
    total_var = np.sum(np.square(targets_orig - target_mean), axis=0)
    residual_var = np.sum(sq_error, axis=0)
    r2_per_target = 1.0 - (residual_var / np.clip(total_var, 1e-12, None))

    variance_orig = np.square(np.clip(sigma_orig, 1e-6, None))
    nll = 0.5 * (
        np.log(2.0 * np.pi * variance_orig)
        + (np.square(targets_orig - mu_orig) / variance_orig)
    )

    metrics = {
        "rmse": float(np.sqrt(np.mean(sq_error))),
        "mae": float(np.mean(abs_error)),
        "r2": float(np.mean(r2_per_target)),
        "mean_nll": float(np.mean(nll)),
    }
    for index, name in enumerate(TARGET_COLUMNS):
        lower = name.lower()
        metrics[f"rmse_{lower}"] = float(rmse_per_target[index])
        metrics[f"mae_{lower}"] = float(mae_per_target[index])
        metrics[f"r2_{lower}"] = float(r2_per_target[index])
    return metrics


def _inverse_target_transform(
    *,
    mu_std: np.ndarray,
    sigma_std: np.ndarray,
    target_scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    mu_orig = target_scaler.inverse_transform(mu_std)
    sigma_orig = sigma_std * target_scaler.scale_[None, :]
    return mu_orig, sigma_orig


def _collect_predictions(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    mu_batches: list[np.ndarray] = []
    sigma_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    with torch.no_grad():
        for feature_batch, target_batch in loader:
            feature_batch = feature_batch.to(device)
            outputs: DESSOutputs = model(feature_batch)
            mu_batches.append(outputs.mu.cpu().numpy())
            sigma_batches.append(outputs.sigma.cpu().numpy())
            target_batches.append(target_batch.cpu().numpy())
    return (
        np.concatenate(mu_batches, axis=0),
        np.concatenate(sigma_batches, axis=0),
        np.concatenate(target_batches, axis=0),
    )


def _evaluate_loader(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scaler: StandardScaler,
) -> dict[str, float]:
    mu_std, sigma_std, target_std = _collect_predictions(model=model, loader=loader, device=device)
    mu_orig, sigma_orig = _inverse_target_transform(
        mu_std=mu_std,
        sigma_std=sigma_std,
        target_scaler=target_scaler,
    )
    target_orig = target_scaler.inverse_transform(target_std)
    return _target_metrics(
        targets_orig=target_orig,
        mu_orig=mu_orig,
        sigma_orig=sigma_orig,
    )


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    config: SingleTargetConfig,
    device: torch.device,
) -> dict[str, float]:
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_mu = 0.0
    total_sigma = 0.0
    total_rows = 0

    for feature_batch, target_batch in loader:
        feature_batch = feature_batch.to(device)
        target_batch = target_batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        outputs: DESSOutputs = model(feature_batch)
        combined, mu_loss, sigma_loss = dess_loss_from_parts(
            outputs.mu,
            outputs.sigma,
            target_batch,
            beta=config.beta,
            alpha=config.alpha,
            reduction="none",
            mu_for_sigma=outputs.mu_for_sigma,
        )
        loss = combined.mean()
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        batch_rows = int(feature_batch.shape[0])
        total_rows += batch_rows
        total_loss += float(loss.detach()) * batch_rows
        total_mu += float(mu_loss.mean().detach()) * batch_rows
        total_sigma += float(sigma_loss.mean().detach()) * batch_rows

    denom = max(total_rows, 1)
    return {
        "loss": total_loss / denom,
        "mu_loss": total_mu / denom,
        "sigma_loss": total_sigma / denom,
    }


def _build_loaders(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        _ArrayDataset(train_features, train_targets),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        _ArrayDataset(val_features, val_targets),
        batch_size=eval_batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        _ArrayDataset(test_features, test_targets),
        batch_size=eval_batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def train_single_target_variant(
    *,
    data_root: Path,
    output_dir: Path,
    config: SingleTargetConfig,
) -> dict[str, float | int | str]:
    _set_seed(config.seed)
    paths = GasTurbinePaths(data_root)
    splits = build_gas_turbine_splits(paths, val_fraction=config.val_fraction)
    train_frame = splits["train"]
    val_frame = splits["val"]
    test_frame = splits["test"]

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    train_features = feature_scaler.fit_transform(train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    val_features = feature_scaler.transform(val_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    test_features = feature_scaler.transform(test_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32))

    train_targets = target_scaler.fit_transform(train_frame[TARGET_COLUMNS].to_numpy(dtype=np.float32))
    val_targets = target_scaler.transform(val_frame[TARGET_COLUMNS].to_numpy(dtype=np.float32))
    test_targets = target_scaler.transform(test_frame[TARGET_COLUMNS].to_numpy(dtype=np.float32))

    train_loader, val_loader, test_loader = _build_loaders(
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        test_features=test_features,
        test_targets=test_targets,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        seed=config.seed,
    )

    device = torch.device(config.device)
    model = _build_model(
        variant=config.variant,
        input_dim=train_features.shape[1],
        output_dim=train_targets.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        train_features=train_features,
        train_targets=train_targets,
    ).to(device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    best_epoch = 0
    patience_left = config.patience
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            config=config,
            device=device,
        )
        val_loss_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            config=config,
            device=device,
        )
        val_eval_metrics = _evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            target_scaler=target_scaler,
        )
        epoch_row = {
            "epoch": float(epoch + 1),
            "train_loss": train_metrics["loss"],
            "train_mu_loss": train_metrics["mu_loss"],
            "train_sigma_loss": train_metrics["sigma_loss"],
            "val_loss": val_loss_metrics["loss"],
            "val_mu_loss": val_loss_metrics["mu_loss"],
            "val_sigma_loss": val_loss_metrics["sigma_loss"],
            "val_rmse": val_eval_metrics["rmse"],
            "val_mae": val_eval_metrics["mae"],
            "val_r2": val_eval_metrics["r2"],
            "val_mean_nll": val_eval_metrics["mean_nll"],
        }
        history.append(epoch_row)

        if val_eval_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_eval_metrics["rmse"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    test_metrics = _evaluate_loader(
        model=model,
        loader=test_loader,
        device=device,
        target_scaler=target_scaler,
    )
    val_metrics = _evaluate_loader(
        model=model,
        loader=val_loader,
        device=device,
        target_scaler=target_scaler,
    )
    mu_std, sigma_std, target_std = _collect_predictions(model=model, loader=test_loader, device=device)
    mu_orig, sigma_orig = _inverse_target_transform(
        mu_std=mu_std,
        sigma_std=sigma_std,
        target_scaler=target_scaler,
    )
    target_orig = target_scaler.inverse_transform(target_std)

    run_name = f"gas_turbine_{config.variant}_seed{config.seed}"
    ensure_dir(output_dir)
    predictions_path = output_dir / f"{run_name}.predictions.csv"
    metrics_path = output_dir / f"{run_name}.metrics.json"
    history_path = output_dir / f"{run_name}.history.csv"
    checkpoint_path = output_dir / f"{run_name}.pt"
    metadata_path = output_dir / f"{run_name}.metadata.json"
    summary_path = output_dir / "summary.csv"

    predictions = test_frame[["year", *FEATURE_COLUMNS, *TARGET_COLUMNS]].copy().reset_index(drop=True)
    predictions["mu_co"] = mu_orig[:, 0]
    predictions["mu_nox"] = mu_orig[:, 1]
    predictions["sigma_co"] = sigma_orig[:, 0]
    predictions["sigma_nox"] = sigma_orig[:, 1]
    predictions.to_csv(predictions_path, index=False)

    metrics_payload = {
        "name": run_name,
        "dataset": "uci_gas_turbine_co_nox",
        "variant": config.variant,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "train_rows": int(train_frame.shape[0]),
        "val_rows": int(val_frame.shape[0]),
        "test_rows": int(test_frame.shape[0]),
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    write_json(metrics_path, metrics_payload)
    pd.DataFrame(history).to_csv(history_path, index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": _jsonable_config(config),
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "feature_scaler_mean": feature_scaler.mean_.tolist(),
            "feature_scaler_scale": feature_scaler.scale_.tolist(),
            "target_scaler_mean": target_scaler.mean_.tolist(),
            "target_scaler_scale": target_scaler.scale_.tolist(),
        },
        checkpoint_path,
    )
    write_json(
        metadata_path,
        {
            "name": run_name,
            "dataset": "uci_gas_turbine_co_nox",
            "dataset_url": GAS_TURBINE_URL,
            "dataset_citation": GAS_TURBINE_CITATION,
            "variant": config.variant,
            "variant_description": SINGLE_TARGET_VARIANT_DESCRIPTIONS[config.variant],
            "config": _jsonable_config(config),
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "split_protocol": (
                "Chronological split following the UCI recommendation: first three years as the training/cross-validation pool, "
                "last two years as test; the last validation fraction of the train pool is used as validation."
            ),
            "train_rows": int(train_frame.shape[0]),
            "val_rows": int(val_frame.shape[0]),
            "test_rows": int(test_frame.shape[0]),
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "predictions_path": str(predictions_path),
            "metrics_path": str(metrics_path),
            "history_path": str(history_path),
            "checkpoint_path": str(checkpoint_path),
        },
    )
    pd.DataFrame([metrics_payload]).to_csv(summary_path, index=False)
    return metrics_payload
