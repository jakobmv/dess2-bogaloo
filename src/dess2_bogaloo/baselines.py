from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dess2_bogaloo.data import DatasetPaths, build_training_subset, embedding_matrix, load_embedding_table
from dess2_bogaloo.utils import cosine_similarity_rows, ensure_dir, l2_normalize


@dataclass(frozen=True)
class RunResult:
    name: str
    frame: pd.DataFrame
    metadata: dict[str, float | int | str]


def _base_run_frame(subset: pd.DataFrame) -> pd.DataFrame:
    return subset[
        ["query_id", "query", "product_id", "product_title", "esci_label"]
    ].copy()


def random_baseline(subset: pd.DataFrame, *, seed: int = 42) -> RunResult:
    rng = np.random.default_rng(seed)
    run = _base_run_frame(subset)
    run["score"] = rng.random(run.shape[0], dtype=np.float32)
    return RunResult(
        name="random",
        frame=run,
        metadata={"seed": seed},
    )


def _vector_similarity_run(
    subset: pd.DataFrame,
    *,
    name: str,
    query_vectors: pd.DataFrame,
    product_vectors: pd.DataFrame,
) -> RunResult:
    run = _base_run_frame(subset)
    query_ids, query_matrix = embedding_matrix(query_vectors, id_col="query_id")
    product_ids, product_matrix = embedding_matrix(product_vectors, id_col="product_id")
    query_lookup = {value: index for index, value in enumerate(query_ids)}
    product_lookup = {value: index for index, value in enumerate(product_ids)}

    query_rows = run["query_id"].map(query_lookup)
    product_rows = run["product_id"].map(product_lookup)
    if query_rows.isna().any():
        missing = run.loc[query_rows.isna(), "query_id"].unique().tolist()[:5]
        raise KeyError(f"Missing query embeddings for query_ids like {missing}")
    if product_rows.isna().any():
        missing = run.loc[product_rows.isna(), "product_id"].unique().tolist()[:5]
        raise KeyError(f"Missing product embeddings for product_ids like {missing}")

    run["score"] = cosine_similarity_rows(
        query_matrix[query_rows.to_numpy(dtype=int)],
        product_matrix[product_rows.to_numpy(dtype=int)],
    )
    return RunResult(name=name, frame=run, metadata={})


def clip_text_baseline(subset: pd.DataFrame, paths: DatasetPaths) -> RunResult:
    query_vectors = load_embedding_table(
        paths.query_features_path,
        id_col="query_id",
        preferred_tokens=["query_embedding", "embedding", "features"],
    )
    product_vectors = load_embedding_table(
        paths.product_features_path,
        id_col="product_id",
        preferred_tokens=[
            "text_embedding",
            "title_embedding",
            "product_title_embedding",
            "text_features",
            "title_features",
        ],
    )
    return _vector_similarity_run(
        subset,
        name="clip_text",
        query_vectors=query_vectors,
        product_vectors=product_vectors,
    )


def clip_image_baseline(subset: pd.DataFrame, paths: DatasetPaths) -> RunResult:
    query_vectors = load_embedding_table(
        paths.query_features_path,
        id_col="query_id",
        preferred_tokens=["query_embedding", "embedding", "features"],
    )
    product_vectors = load_embedding_table(
        paths.product_features_path,
        id_col="product_id",
        preferred_tokens=[
            "image_embedding",
            "image_features",
            "product_image_embedding",
            "visual_embedding",
            "visual_features",
            "embedding",
            "features",
        ],
    )
    return _vector_similarity_run(
        subset,
        name="clip_image",
        query_vectors=query_vectors,
        product_vectors=product_vectors,
    )


def _load_or_encode_texts(
    *,
    model_name: str,
    texts: list[str],
    ids: list[str],
    cache_path: Path,
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    ensure_dir(cache_path.parent)
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        return pd.DataFrame(
            {
                "id": payload["ids"],
                "embedding": list(payload["embeddings"]),
            }
        )

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.savez_compressed(cache_path, ids=np.asarray(ids), embeddings=embeddings)
    return pd.DataFrame({"id": ids, "embedding": list(embeddings)})


def _make_embedding_frame(
    *,
    ids: list[str],
    embeddings: np.ndarray,
    id_name: str,
) -> pd.DataFrame:
    frame = pd.DataFrame({id_name: ids, "embedding": list(embeddings.astype(np.float32))})
    frame.attrs["embedding_columns"] = ["embedding"]
    frame.attrs["is_vector_column"] = True
    return frame


def sbert_text_baseline(
    subset: pd.DataFrame,
    *,
    cache_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
) -> RunResult:
    from sentence_transformers import SentenceTransformer

    query_table = subset[["query_id", "query"]].drop_duplicates().sort_values("query_id")
    product_table = subset[["product_id", "product_title"]].drop_duplicates().sort_values("product_id")

    query_cache = cache_dir / "sbert_text_queries.npz"
    product_cache = cache_dir / "sbert_text_products.npz"
    if query_cache.exists() and product_cache.exists():
        query_embeddings = _load_or_encode_texts(
            model_name=model_name,
            texts=[],
            ids=[],
            cache_path=query_cache,
        ).rename(columns={"id": "query_id"})
        product_embeddings = _load_or_encode_texts(
            model_name=model_name,
            texts=[],
            ids=[],
            cache_path=product_cache,
        ).rename(columns={"id": "product_id"})
    else:
        model = SentenceTransformer(model_name)
        query_vectors = model.encode(
            query_table["query"].tolist(),
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
        product_vectors = model.encode(
            product_table["product_title"].fillna("").tolist(),
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
        ensure_dir(cache_dir)
        np.savez_compressed(
            query_cache,
            ids=query_table["query_id"].astype(str).to_numpy(),
            embeddings=query_vectors,
        )
        np.savez_compressed(
            product_cache,
            ids=product_table["product_id"].astype(str).to_numpy(),
            embeddings=product_vectors,
        )
        query_embeddings = _make_embedding_frame(
            ids=query_table["query_id"].astype(str).tolist(),
            embeddings=query_vectors,
            id_name="query_id",
        )
        product_embeddings = _make_embedding_frame(
            ids=product_table["product_id"].astype(str).tolist(),
            embeddings=product_vectors,
            id_name="product_id",
        )

    query_embeddings["query_id"] = query_embeddings["query_id"].astype(query_table["query_id"].dtype)
    product_embeddings["product_id"] = product_embeddings["product_id"].astype(product_table["product_id"].dtype)
    query_embeddings.attrs["embedding_columns"] = ["embedding"]
    query_embeddings.attrs["is_vector_column"] = True
    product_embeddings.attrs["embedding_columns"] = ["embedding"]
    product_embeddings.attrs["is_vector_column"] = True
    return _vector_similarity_run(
        subset,
        name="sbert_text",
        query_vectors=query_embeddings,
        product_vectors=product_embeddings,
    )


def combine_runs(
    left: RunResult,
    right: RunResult,
    *,
    alpha: float,
    method: str,
) -> RunResult:
    merged = left.frame[["query_id", "product_id", "score"]].merge(
        right.frame[["query_id", "product_id", "score"]],
        on=["query_id", "product_id"],
        suffixes=("_left", "_right"),
    )
    if method == "score":
        merged["score"] = alpha * merged["score_left"] + (1.0 - alpha) * merged["score_right"]
    elif method == "rank":
        merged["rank_left"] = (
            merged.groupby("query_id")["score_left"]
            .rank(method="first", ascending=False)
        )
        merged["rank_right"] = (
            merged.groupby("query_id")["score_right"]
            .rank(method="first", ascending=False)
        )
        merged["score"] = -(alpha * merged["rank_left"] + (1.0 - alpha) * merged["rank_right"])
    else:
        raise ValueError(f"Unsupported fusion method: {method}")

    frame = left.frame.drop(columns=["score"]).merge(
        merged[["query_id", "product_id", "score"]],
        on=["query_id", "product_id"],
        how="inner",
    )
    return RunResult(
        name=f"{left.name}_{right.name}_{method}_a{alpha:.2f}",
        frame=frame,
        metadata={"alpha": alpha, "method": method},
    )


def train_esci_baseline_model(
    *,
    paths: DatasetPaths,
    model_dir: Path,
    random_state: int = 42,
    n_dev_queries: int = 400,
    train_batch_size: int = 32,
    max_train_rows: int | None = None,
) -> Path:
    import torch
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
    from torch.utils.data import DataLoader

    ensure_dir(model_dir.parent)
    if model_dir.exists():
        return model_dir

    train = build_training_subset(paths)
    query_ids = train["query_id"].unique()
    dev_size = min(max(n_dev_queries / len(query_ids), 0.0), 0.25)
    train_ids, dev_ids = train_test_split(query_ids, test_size=dev_size, random_state=random_state)
    train_frame = train.loc[train["query_id"].isin(train_ids), ["query", "product_title", "gain"]]
    dev_frame = train.loc[train["query_id"].isin(dev_ids), ["query", "product_title", "gain"]]
    if max_train_rows is not None:
        train_frame = train_frame.head(max_train_rows).copy()

    samples = [
        InputExample(texts=[row.query, row.product_title], label=float(row.gain))
        for row in train_frame.itertuples(index=False)
    ]
    loader = DataLoader(samples, shuffle=True, batch_size=train_batch_size, drop_last=True)

    dev_samples: dict[int, dict[str, str | list[str]]] = {}
    query_lookup: dict[str, int] = {}
    for row in dev_frame.itertuples(index=False):
        qid = query_lookup.setdefault(row.query, len(query_lookup))
        if qid not in dev_samples:
            dev_samples[qid] = {"query": row.query, "positive": [], "negative": []}
        if row.gain > 0:
            if row.product_title not in dev_samples[qid]["positive"]:
                dev_samples[qid]["positive"].append(row.product_title)
        else:
            if row.product_title not in dev_samples[qid]["negative"]:
                dev_samples[qid]["negative"].append(row.product_title)
    dev_samples = {
        qid: sample
        for qid, sample in dev_samples.items()
        if len(sample["positive"]) > 0
        and len(sample["negative"]) > 0
        and (len(sample["positive"]) + len(sample["negative"])) > 1
    }

    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        num_labels=1,
        max_length=512,
        activation_fn=torch.nn.Identity(),
    )
    evaluator = CERerankingEvaluator(dev_samples, name="esci-dev") if dev_samples else None
    model.old_fit(
        train_dataloader=loader,
        loss_fct=torch.nn.MSELoss(),
        evaluator=evaluator,
        epochs=1,
        evaluation_steps=5000,
        warmup_steps=5000,
        output_path=str(model_dir),
        optimizer_params={"lr": 7e-6},
    )
    return model_dir


def score_esci_baseline(
    subset: pd.DataFrame,
    *,
    model_dir: Path,
    batch_size: int = 256,
) -> RunResult:
    from sentence_transformers.cross_encoder import CrossEncoder

    run = _base_run_frame(subset)
    model = CrossEncoder(str(model_dir))
    pairs = list(zip(run["query"].tolist(), run["product_title"].fillna("").tolist()))
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    run["score"] = np.asarray(scores, dtype=np.float32)
    return RunResult(name="esci_baseline", frame=run, metadata={"model_dir": str(model_dir)})
