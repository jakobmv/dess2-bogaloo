from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dess2_bogaloo.data import DatasetPaths, build_reranking_subset
from dess2_bogaloo.dess_model import DESSOutputs, VARIANT_MODEL_TYPES
from dess2_bogaloo.eval import evaluate_run
from dess2_bogaloo.train import _load_feature_tables
from dess2_bogaloo.utils import ensure_dir, l2_normalize, write_json


@dataclass(frozen=True)
class DessSamplingConfig:
    seed: int = 42
    device: str = "cpu"
    eval_batch_size: int = 2048
    model_name: str = "sentence-transformers/all-MiniLM-L12-v2"
    cache_dir: Path = Path("outputs/reproduction/cache")


def _stable_query_seed(seed: int, query_id: object) -> int:
    digest = hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def load_dess_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload["config"]
    variant = str(config["variant"])
    model_type = VARIANT_MODEL_TYPES[variant]
    model = model_type(
        input_dim=int(payload["input_dim"]),
        output_dim=int(payload["output_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def predict_query_distributions(
    *,
    model: torch.nn.Module,
    query_table: pd.DataFrame,
    query_matrix: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[object, np.ndarray], dict[object, np.ndarray]]:
    mu_batches: list[np.ndarray] = []
    sigma_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, query_matrix.shape[0], batch_size):
            stop = min(start + batch_size, query_matrix.shape[0])
            query_batch = torch.from_numpy(query_matrix[start:stop]).to(device)
            outputs: DESSOutputs = model(query_batch)
            mu_batches.append(outputs.mu.cpu().numpy())
            sigma_batches.append(outputs.sigma.cpu().numpy())

    mu_matrix = np.concatenate(mu_batches, axis=0)
    sigma_matrix = np.concatenate(sigma_batches, axis=0)
    query_ids = query_table["query_id"].tolist()
    mu_lookup = {query_id: mu_matrix[index] for index, query_id in enumerate(query_ids)}
    sigma_lookup = {query_id: sigma_matrix[index] for index, query_id in enumerate(query_ids)}
    return mu_lookup, sigma_lookup


def sample_candidate_order(
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    candidate_matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    num_candidates = int(candidate_matrix.shape[0])
    if num_candidates == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    draw_order = np.empty(num_candidates, dtype=np.int64)
    draw_cosine = np.empty(num_candidates, dtype=np.float32)
    normalized_candidates = l2_normalize(candidate_matrix.astype(np.float32, copy=False))
    sampled_points = rng.normal(
        loc=mu,
        scale=sigma,
        size=(num_candidates, mu.shape[0]),
    ).astype(np.float32, copy=False)
    normalized_samples = l2_normalize(sampled_points)
    similarity_matrix = normalized_samples @ normalized_candidates.T
    remaining_mask = np.ones(num_candidates, dtype=bool)

    for step in range(num_candidates):
        similarities = similarity_matrix[step].copy()
        similarities[~remaining_mask] = -np.inf
        chosen_row = int(np.argmax(similarities))
        remaining_mask[chosen_row] = False
        draw_order[chosen_row] = step + 1
        draw_cosine[chosen_row] = float(similarities[chosen_row])

    return draw_order, draw_cosine


def sample_rerank_subset(
    subset: pd.DataFrame,
    *,
    mu_lookup: dict[object, np.ndarray],
    sigma_lookup: dict[object, np.ndarray],
    product_table: pd.DataFrame,
    product_matrix: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    run = subset[["query_id", "query", "product_id", "product_title", "esci_label"]].copy()
    num_rows = int(run.shape[0])
    score_values = np.empty(num_rows, dtype=np.float32)
    draw_order_values = np.empty(num_rows, dtype=np.int64)
    draw_cosine_values = np.empty(num_rows, dtype=np.float32)
    product_lookup = {value: index for index, value in enumerate(product_table["product_id"].tolist())}
    product_ids = run["product_id"].to_numpy()
    for query_id, group_rows in run.groupby("query_id", sort=False).indices.items():
        row_indices = np.asarray(group_rows, dtype=np.int64)
        try:
            product_rows = np.asarray([product_lookup[product_ids[index]] for index in row_indices], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"Missing product embedding for product_id {exc.args[0]}") from exc
        candidate_matrix = product_matrix[product_rows]
        rng = np.random.default_rng(_stable_query_seed(seed, query_id))
        draw_order, draw_cosine = sample_candidate_order(
            mu=mu_lookup[query_id],
            sigma=sigma_lookup[query_id],
            candidate_matrix=candidate_matrix,
            rng=rng,
        )
        score_values[row_indices] = (row_indices.shape[0] - draw_order).astype(np.float32)
        draw_order_values[row_indices] = draw_order
        draw_cosine_values[row_indices] = draw_cosine

    run["score"] = score_values
    run["draw_order"] = draw_order_values
    run["draw_cosine"] = draw_cosine_values
    return run


def run_dess_sampling_reranker(
    *,
    paths: DatasetPaths,
    checkpoint_path: Path,
    output_dir: Path,
    config: DessSamplingConfig,
) -> dict[str, float | int | str]:
    ensure_dir(output_dir)
    device = torch.device(config.device)
    model, payload = load_dess_checkpoint(checkpoint_path, device=device)
    checkpoint_config = payload["config"]
    variant = str(checkpoint_config["variant"])
    model_name = str(checkpoint_config.get("model_name", config.model_name))
    feature_source = str(checkpoint_config.get("feature_source", "sbert_text"))
    eval_subset = build_reranking_subset(paths)
    query_table, query_matrix, product_table, product_matrix = _load_feature_tables(
        eval_subset,
        paths=paths,
        feature_source=feature_source,
        cache_dir=config.cache_dir,
        prefix=feature_source,
        model_name=model_name,
    )
    mu_lookup, sigma_lookup = predict_query_distributions(
        model=model,
        query_table=query_table,
        query_matrix=query_matrix,
        batch_size=config.eval_batch_size,
        device=device,
    )
    run = sample_rerank_subset(
        eval_subset,
        mu_lookup=mu_lookup,
        sigma_lookup=sigma_lookup,
        product_table=product_table,
        product_matrix=product_matrix,
        seed=config.seed,
    )
    metrics = evaluate_run(run)

    run_name = f"dess_sampling_{feature_source}_{variant}_seed{config.seed}"
    run_path = output_dir / f"{run_name}.csv"
    metrics_path = output_dir / f"{run_name}.metrics.json"
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / f"{run_name}.metadata.json"

    run.to_csv(run_path, index=False)
    write_json(metrics_path, metrics)
    pd.DataFrame([{"name": run_name, **metrics}]).to_csv(summary_path, index=False)
    write_json(
        metadata_path,
        {
            "name": run_name,
            "variant": variant,
            "feature_source": feature_source,
            "seed": config.seed,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in checkpoint_config.items()
            },
            "sampling_method": (
                "For each query, repeatedly sample z ~ N(mu, sigma^2) independently per dimension; "
                "choose the remaining candidate with highest cosine similarity to z; remove it; repeat "
                "until the candidate list is exhausted."
            ),
            "run_path": str(run_path),
            "metrics_path": str(metrics_path),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
        },
    )
    return {
        "name": run_name,
        "variant": variant,
        "seed": int(config.seed),
        "ndcg": float(metrics["ndcg"]),
        "num_queries": int(metrics["num_queries"]),
        "num_judgements": int(metrics["num_judgements"]),
        "checkpoint_path": str(checkpoint_path),
    }
