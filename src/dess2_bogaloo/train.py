from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from dess2_bogaloo.data import (
    DatasetPaths,
    build_reranking_subset,
    build_training_subset,
    embedding_matrix,
    load_embedding_table,
)
from dess2_bogaloo.dess_model import VARIANT_MODEL_TYPES, DESSOutputs, TextDessAdapter
from dess2_bogaloo.dess_original import F_dess_loss as original_dess_loss
from dess2_bogaloo.dess_updated import dess_loss_from_parts, gaussian_log_score
from dess2_bogaloo.eval import evaluate_run
from dess2_bogaloo.utils import ensure_dir, write_json


DESS_FEATURE_SOURCES = ("sbert_text", "clip_image")


def ensure_reproduction_verified(summary_path: Path) -> None:
    if not summary_path.exists():
        raise FileNotFoundError(
            "Reproduction summary not found. Per AGENTS.md, DESS work starts only after reproduction is verified."
        )


def _jsonable_config(config: DessTrainConfig) -> dict[str, object]:
    payload = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in payload.items()
    }


@dataclass(frozen=True)
class DessTrainConfig:
    batch_size: int = 256
    eval_batch_size: int = 2048
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta: float = 1.0
    alpha: float = 0.5
    dropout: float = 0.1
    hidden_dim: int = 1024
    min_gain: float = 0.01
    max_train_rows: int | None = 50000
    seed: int = 42
    device: str = "cpu"
    model_name: str = "sentence-transformers/all-MiniLM-L12-v2"
    cache_dir: Path = Path("outputs/reproduction/cache")
    variant: str = "mlp_joint"
    feature_source: str = "sbert_text"


class _PairDataset(Dataset):
    def __init__(
        self,
        query_vectors: np.ndarray,
        product_vectors: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        self.query_vectors = torch.from_numpy(query_vectors.astype(np.float32, copy=False))
        self.product_vectors = torch.from_numpy(product_vectors.astype(np.float32, copy=False))
        self.weights = torch.from_numpy(weights.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.query_vectors.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.query_vectors[index],
            self.product_vectors[index],
            self.weights[index],
        )


def _load_embedding_frame(cache_path: Path, *, id_col: str) -> pd.DataFrame:
    payload = np.load(cache_path, allow_pickle=True)
    frame = pd.DataFrame(
        {
            id_col: payload["ids"],
            "embedding": list(payload["embeddings"]),
        }
    )
    return frame


def _save_embedding_frame(
    cache_path: Path,
    *,
    ids: list[str],
    embeddings: np.ndarray,
) -> None:
    ensure_dir(cache_path.parent)
    np.savez_compressed(
        cache_path,
        ids=np.asarray(ids),
        embeddings=embeddings.astype(np.float32),
    )


def _encode_unique_texts(
    *,
    frame: pd.DataFrame,
    id_col: str,
    text_col: str,
    cache_path: Path,
    model_name: str,
) -> pd.DataFrame:
    unique = frame[[id_col, text_col]].drop_duplicates(subset=[id_col]).copy()
    unique[id_col] = unique[id_col].astype(str)
    unique[text_col] = unique[text_col].fillna("")
    cached: pd.DataFrame | None = None
    cached_ids: set[str] = set()
    if cache_path.exists():
        cached = _load_embedding_frame(cache_path, id_col=id_col)
        cached[id_col] = cached[id_col].astype(str)
        cached_ids = set(cached[id_col].tolist())

    missing = unique.loc[~unique[id_col].isin(cached_ids)].copy()
    if not missing.empty:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cpu")
        embeddings = model.encode(
            missing[text_col].tolist(),
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
        new_rows = pd.DataFrame({id_col: missing[id_col].tolist(), "embedding": list(embeddings)})
        cached = new_rows if cached is None else pd.concat([cached, new_rows], ignore_index=True)
        _save_embedding_frame(
            cache_path,
            ids=cached[id_col].tolist(),
            embeddings=np.stack(cached["embedding"].map(lambda value: np.asarray(value, dtype=np.float32))),
        )

    if cached is None:
        raise RuntimeError(f"Failed to build embedding cache at {cache_path}")

    selected = unique[[id_col]].merge(cached, on=id_col, how="left")
    if selected["embedding"].isna().any():
        missing_ids = selected.loc[selected["embedding"].isna(), id_col].tolist()[:5]
        raise KeyError(f"Embedding cache {cache_path} is missing ids like {missing_ids}")
    return selected


def _load_sbert_tables(
    subset: pd.DataFrame,
    *,
    cache_dir: Path,
    prefix: str,
    model_name: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    query_cache = cache_dir / f"{prefix}_queries.npz"
    product_cache = cache_dir / f"{prefix}_products.npz"

    query_table = _encode_unique_texts(
        frame=subset[["query_id", "query"]],
        id_col="query_id",
        text_col="query",
        cache_path=query_cache,
        model_name=model_name,
    )
    product_table = _encode_unique_texts(
        frame=subset[["product_id", "product_title"]],
        id_col="product_id",
        text_col="product_title",
        cache_path=product_cache,
        model_name=model_name,
    )
    query_table["query_id"] = query_table["query_id"].astype(subset["query_id"].dtype)
    product_table["product_id"] = product_table["product_id"].astype(subset["product_id"].dtype)

    query_matrix = np.stack(query_table["embedding"].map(lambda value: np.asarray(value, dtype=np.float32)))
    product_matrix = np.stack(product_table["embedding"].map(lambda value: np.asarray(value, dtype=np.float32)))
    return query_table, query_matrix, product_table, product_matrix


def _load_frozen_feature_tables(
    subset: pd.DataFrame,
    *,
    paths: DatasetPaths,
    query_tokens: list[str],
    product_tokens: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    expected_query_ids = subset["query_id"].drop_duplicates()
    expected_product_ids = subset["product_id"].drop_duplicates()
    query_frame = load_embedding_table(
        paths.query_features_path,
        id_col="query_id",
        preferred_tokens=query_tokens,
    )
    product_frame = load_embedding_table(
        paths.product_features_path,
        id_col="product_id",
        preferred_tokens=product_tokens,
    )
    query_frame = query_frame.loc[query_frame["query_id"].isin(expected_query_ids)].copy()
    product_frame = product_frame.loc[product_frame["product_id"].isin(expected_product_ids)].copy()
    missing_query_ids = expected_query_ids.loc[~expected_query_ids.isin(query_frame["query_id"])].tolist()
    missing_product_ids = expected_product_ids.loc[
        ~expected_product_ids.isin(product_frame["product_id"])
    ].tolist()
    if missing_query_ids:
        sample = missing_query_ids[:5]
        raise KeyError(
            "Frozen feature table is missing query embeddings for "
            f"{len(missing_query_ids)} ids, including {sample}. "
            "For SQID CLIP features, the released query_features.parquet only covers the reranking test queries."
        )
    if missing_product_ids:
        sample = missing_product_ids[:5]
        raise KeyError(
            "Frozen feature table is missing product embeddings for "
            f"{len(missing_product_ids)} ids, including {sample}."
        )
    query_frame["query_id"] = query_frame["query_id"].astype(subset["query_id"].dtype)
    product_frame["product_id"] = product_frame["product_id"].astype(subset["product_id"].dtype)
    query_frame = query_frame.sort_values("query_id", kind="mergesort").reset_index(drop=True)
    product_frame = product_frame.sort_values("product_id", kind="mergesort").reset_index(drop=True)
    query_ids, query_matrix = embedding_matrix(query_frame, id_col="query_id")
    product_ids, product_matrix = embedding_matrix(product_frame, id_col="product_id")
    query_table = pd.DataFrame({"query_id": query_ids, "embedding": list(query_matrix)})
    product_table = pd.DataFrame({"product_id": product_ids, "embedding": list(product_matrix)})
    return query_table, query_matrix, product_table, product_matrix


def _load_feature_tables(
    subset: pd.DataFrame,
    *,
    paths: DatasetPaths,
    feature_source: str,
    cache_dir: Path,
    prefix: str,
    model_name: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    if feature_source == "sbert_text":
        return _load_sbert_tables(
            subset,
            cache_dir=cache_dir,
            prefix=prefix,
            model_name=model_name,
        )
    if feature_source == "clip_image":
        return _load_frozen_feature_tables(
            subset,
            paths=paths,
            query_tokens=[
                "clip_text_features",
                "query_embedding",
                "embedding",
                "features",
            ],
            product_tokens=[
                "clip_image_features",
                "image_embedding",
                "image_features",
                "product_image_embedding",
                "visual_embedding",
                "visual_features",
                "embedding",
                "features",
            ],
        )
    raise ValueError(
        f"Unsupported feature source: {feature_source}. Expected one of {DESS_FEATURE_SOURCES}."
    )


def _positive_training_subset(
    paths: DatasetPaths,
    *,
    min_gain: float,
    max_train_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    subset = build_training_subset(paths)
    subset = subset.loc[subset["gain"] >= min_gain].copy()
    subset = subset.sort_values(["query_id", "product_id"], kind="mergesort").reset_index(drop=True)
    if max_train_rows is not None and subset.shape[0] > max_train_rows:
        subset = subset.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)
    return subset


def _align_pair_arrays(
    subset: pd.DataFrame,
    *,
    query_table: pd.DataFrame,
    query_matrix: np.ndarray,
    product_table: pd.DataFrame,
    product_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_lookup = {value: index for index, value in enumerate(query_table["query_id"].tolist())}
    product_lookup = {value: index for index, value in enumerate(product_table["product_id"].tolist())}

    query_rows = subset["query_id"].map(query_lookup)
    product_rows = subset["product_id"].map(product_lookup)
    if query_rows.isna().any():
        missing = subset.loc[query_rows.isna(), "query_id"].unique().tolist()[:5]
        raise KeyError(f"Missing query embeddings for query_ids like {missing}")
    if product_rows.isna().any():
        missing = subset.loc[product_rows.isna(), "product_id"].unique().tolist()[:5]
        raise KeyError(f"Missing product embeddings for product_ids like {missing}")

    queries = query_matrix[query_rows.to_numpy(dtype=int)]
    products = product_matrix[product_rows.to_numpy(dtype=int)]
    weights = subset["gain"].to_numpy(dtype=np.float32, copy=True)
    return queries, products, weights


def probe_original_multi_target_loss(
    subset: pd.DataFrame,
    *,
    product_table: pd.DataFrame,
    product_matrix: np.ndarray,
) -> dict[str, str | bool]:
    grouped = subset.groupby("query_id", sort=False)
    product_lookup = {value: index for index, value in enumerate(product_table["product_id"].tolist())}

    targets: list[np.ndarray] = []
    for _, group in grouped:
        if group.shape[0] < 2:
            continue
        product_rows = group["product_id"].map(product_lookup).to_numpy(dtype=int)[:2]
        target = product_matrix[product_rows].T
        targets.append(target.astype(np.float32, copy=False))
        if len(targets) == 2:
            break

    if len(targets) < 2:
        return {"ok": False, "reason": "Not enough multi-target positive queries available for a probe batch."}

    embedding_dim = targets[0].shape[0]
    target_batch = torch.from_numpy(np.stack(targets))
    pred = torch.randn(target_batch.shape[0], embedding_dim * 2)
    try:
        original_dess_loss(pred, target_batch, reduction="mean")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "reason": "Official multi-target loss accepted a grouped SBERT batch."}


def _train_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    config: DessTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mu = 0.0
    total_sigma = 0.0
    total_examples = 0

    for query_batch, product_batch, weight_batch in loader:
        query_batch = query_batch.to(device)
        product_batch = product_batch.to(device)
        weight_batch = weight_batch.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        outputs: DESSOutputs = model(query_batch)
        combined, mu_loss, sigma_loss = dess_loss_from_parts(
            outputs.mu,
            outputs.sigma,
            product_batch,
            beta=config.beta,
            alpha=config.alpha,
            reduction="none",
            mu_for_sigma=outputs.mu_for_sigma,
        )
        per_example = combined.mean(dim=-1)
        weighted_loss = (per_example * weight_batch).sum() / weight_batch.sum().clamp_min(1e-6)
        if optimizer is not None:
            weighted_loss.backward()
            optimizer.step()

        batch_size = int(query_batch.shape[0])
        total_loss += float(weighted_loss.detach()) * batch_size
        total_mu += float(mu_loss.mean().detach()) * batch_size
        total_sigma += float(sigma_loss.mean().detach()) * batch_size
        total_examples += batch_size

    denom = max(total_examples, 1)
    return {
        "loss": total_loss / denom,
        "mu_loss": total_mu / denom,
        "sigma_loss": total_sigma / denom,
    }


def _score_subset(
    subset: pd.DataFrame,
    *,
    model: torch.nn.Module,
    query_table: pd.DataFrame,
    query_matrix: np.ndarray,
    product_table: pd.DataFrame,
    product_matrix: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    run = subset[["query_id", "query", "product_id", "product_title", "esci_label"]].copy()
    query_lookup = {value: index for index, value in enumerate(query_table["query_id"].tolist())}
    product_lookup = {value: index for index, value in enumerate(product_table["product_id"].tolist())}

    query_rows = run["query_id"].map(query_lookup)
    product_rows = run["product_id"].map(product_lookup)
    if query_rows.isna().any():
        missing = run.loc[query_rows.isna(), "query_id"].unique().tolist()[:5]
        raise KeyError(f"Missing eval query embeddings for query_ids like {missing}")
    if product_rows.isna().any():
        missing = run.loc[product_rows.isna(), "product_id"].unique().tolist()[:5]
        raise KeyError(f"Missing eval product embeddings for product_ids like {missing}")

    scores = np.empty(run.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, run.shape[0], batch_size):
            stop = min(start + batch_size, run.shape[0])
            query_batch = torch.from_numpy(query_matrix[query_rows.iloc[start:stop].to_numpy(dtype=int)]).to(device)
            product_batch = torch.from_numpy(product_matrix[product_rows.iloc[start:stop].to_numpy(dtype=int)]).to(device)
            outputs: DESSOutputs = model(query_batch)
            batch_scores = gaussian_log_score(outputs.mu, outputs.sigma, product_batch)
            scores[start:stop] = batch_scores.cpu().numpy()

    run["score"] = scores
    return run


def train_and_evaluate_dess(
    *,
    paths: DatasetPaths,
    output_dir: Path,
    config: DessTrainConfig,
) -> dict[str, float | int | str]:
    ensure_dir(output_dir)
    ensure_dir(config.cache_dir)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.variant not in VARIANT_MODEL_TYPES:
        raise ValueError(f"Unsupported DESS variant: {config.variant}")
    if config.feature_source not in DESS_FEATURE_SOURCES:
        raise ValueError(
            f"Unsupported feature source: {config.feature_source}. Expected one of {DESS_FEATURE_SOURCES}."
        )

    train_subset = _positive_training_subset(
        paths,
        min_gain=config.min_gain,
        max_train_rows=config.max_train_rows,
        seed=config.seed,
    )
    train_prefix = "dess_sbert_train" if config.feature_source == "sbert_text" else config.feature_source
    train_query_table, train_query_matrix, train_product_table, train_product_matrix = _load_feature_tables(
        train_subset,
        paths=paths,
        feature_source=config.feature_source,
        cache_dir=config.cache_dir,
        prefix=train_prefix,
        model_name=config.model_name,
    )
    original_probe = probe_original_multi_target_loss(
        train_subset,
        product_table=train_product_table,
        product_matrix=train_product_matrix,
    )
    train_queries, train_products, train_weights = _align_pair_arrays(
        train_subset,
        query_table=train_query_table,
        query_matrix=train_query_matrix,
        product_table=train_product_table,
        product_matrix=train_product_matrix,
    )

    dataset = _PairDataset(train_queries, train_products, train_weights)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    device = torch.device(config.device)
    model_type = VARIANT_MODEL_TYPES[config.variant]
    model = model_type(
        input_dim=int(train_queries.shape[1]),
        output_dim=int(train_products.shape[1]),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer: torch.optim.Optimizer | None
    if trainable_params:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = None

    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        metrics = _train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            config=config,
            device=device,
        )
        metrics["epoch"] = float(epoch + 1)
        history.append(metrics)

    eval_subset = build_reranking_subset(paths)
    eval_query_table, eval_query_matrix, eval_product_table, eval_product_matrix = _load_feature_tables(
        eval_subset,
        paths=paths,
        feature_source=config.feature_source,
        cache_dir=config.cache_dir,
        prefix=config.feature_source,
        model_name=config.model_name,
    )
    run = _score_subset(
        eval_subset,
        model=model,
        query_table=eval_query_table,
        query_matrix=eval_query_matrix,
        product_table=eval_product_table,
        product_matrix=eval_product_matrix,
        batch_size=config.eval_batch_size,
        device=device,
    )
    metrics = evaluate_run(run)

    run_name = f"dess_{config.feature_source}_{config.variant}"
    run_path = output_dir / f"{run_name}.csv"
    metrics_path = output_dir / f"{run_name}.metrics.json"
    summary_path = output_dir / "summary.csv"
    checkpoint_path = output_dir / f"{run_name}.pt"
    metadata_path = output_dir / f"{run_name}.metadata.json"
    history_path = output_dir / f"{run_name}.history.csv"

    run.to_csv(run_path, index=False)
    write_json(metrics_path, metrics)
    pd.DataFrame([{"name": run_name, **metrics}]).to_csv(summary_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "input_dim": int(train_queries.shape[1]),
            "output_dim": int(train_products.shape[1]),
        },
        checkpoint_path,
    )
    write_json(
        metadata_path,
        {
            "name": run_name,
            "config": _jsonable_config(config),
            "feature_source": config.feature_source,
            "loss_impl": "dess_updated",
            "variant": config.variant,
            "official_multi_target_probe": original_probe,
            "train_rows": int(train_subset.shape[0]),
            "train_queries": int(train_subset["query_id"].nunique()),
            "history": history,
            "checkpoint_path": str(checkpoint_path),
            "run_path": str(run_path),
            "metrics_path": str(metrics_path),
            "history_path": str(history_path),
        },
    )
    return {
        "name": run_name,
        "ndcg": float(metrics["ndcg"]),
        "num_queries": int(metrics["num_queries"]),
        "num_judgements": int(metrics["num_judgements"]),
        "train_rows": int(train_subset.shape[0]),
        "loss_impl": "dess_updated",
        "variant": config.variant,
        "feature_source": config.feature_source,
    }
