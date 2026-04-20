from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd

from dess2_bogaloo.eval import LABEL_GAINS
from dess2_bogaloo.utils import ensure_dir


ESCI_FILES = {
    "examples": "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
    "products": "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet",
    "sources": "https://raw.githubusercontent.com/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_sources.csv",
}

SQID_FILES = {
    "product_image_urls": "https://raw.githubusercontent.com/Crossing-Minds/shopping-queries-image-dataset/main/sqid/product_image_urls.csv",
    "product_features": "https://media.githubusercontent.com/media/Crossing-Minds/shopping-queries-image-dataset/main/sqid/product_features.parquet",
    "query_features": "https://media.githubusercontent.com/media/Crossing-Minds/shopping-queries-image-dataset/main/sqid/query_features.parquet",
    "supp_product_image_urls": "https://raw.githubusercontent.com/Crossing-Minds/shopping-queries-image-dataset/main/sqid/supp_product_image_urls.csv",
}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def esci_dir(self) -> Path:
        return self.raw_dir / "esci"

    @property
    def sqid_dir(self) -> Path:
        return self.raw_dir / "sqid"

    @property
    def examples_path(self) -> Path:
        return self.esci_dir / "shopping_queries_dataset_examples.parquet"

    @property
    def products_path(self) -> Path:
        return self.esci_dir / "shopping_queries_dataset_products.parquet"

    @property
    def sources_path(self) -> Path:
        return self.esci_dir / "shopping_queries_dataset_sources.csv"

    @property
    def product_image_urls_path(self) -> Path:
        return self.sqid_dir / "product_image_urls.csv"

    @property
    def product_features_path(self) -> Path:
        return self.sqid_dir / "product_features.parquet"

    @property
    def query_features_path(self) -> Path:
        return self.sqid_dir / "query_features.parquet"

    @property
    def supp_product_image_urls_path(self) -> Path:
        return self.sqid_dir / "supp_product_image_urls.csv"


def download_url(url: str, destination: Path, *, chunk_size: int = 1 << 20) -> Path:
    ensure_dir(destination.parent)
    with urlopen(url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
    return destination


def download_required_data(paths: DatasetPaths, *, overwrite: bool = False) -> list[Path]:
    downloaded: list[Path] = []
    targets = {
        paths.examples_path: ESCI_FILES["examples"],
        paths.products_path: ESCI_FILES["products"],
        paths.sources_path: ESCI_FILES["sources"],
        paths.product_image_urls_path: SQID_FILES["product_image_urls"],
        paths.product_features_path: SQID_FILES["product_features"],
        paths.query_features_path: SQID_FILES["query_features"],
        paths.supp_product_image_urls_path: SQID_FILES["supp_product_image_urls"],
    }
    for destination, url in targets.items():
        if destination.exists() and not overwrite:
            continue
        download_url(url, destination)
        downloaded.append(destination)
    return downloaded


def load_examples(paths: DatasetPaths) -> pd.DataFrame:
    return pd.read_parquet(paths.examples_path)


def load_products(paths: DatasetPaths) -> pd.DataFrame:
    return pd.read_parquet(paths.products_path)


def build_reranking_subset(paths: DatasetPaths) -> pd.DataFrame:
    examples = load_examples(paths)
    products = load_products(paths)
    merged = examples.merge(
        products,
        how="left",
        on=["product_locale", "product_id"],
    )
    subset = merged.loc[
        (merged["small_version"] == 1)
        & (merged["split"] == "test")
        & (merged["product_locale"] == "us")
    ].copy()
    subset["gain"] = subset["esci_label"].map(LABEL_GAINS).astype(float)
    return subset


def build_training_subset(paths: DatasetPaths) -> pd.DataFrame:
    examples = load_examples(paths)
    products = load_products(paths)
    merged = examples.merge(
        products,
        how="left",
        on=["product_locale", "product_id"],
    )
    subset = merged.loc[
        (merged["small_version"] == 1)
        & (merged["split"] == "train")
        & (merged["product_locale"] == "us")
    ].copy()
    subset["gain"] = subset["esci_label"].map(LABEL_GAINS).astype(float)
    return subset


def _looks_like_vector(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _sample_non_null(series: pd.Series) -> list[Any]:
    values = [value for value in series.head(20).tolist() if value is not None]
    return values


def infer_embedding_columns(
    frame: pd.DataFrame,
    *,
    id_col: str,
    preferred_tokens: list[str],
) -> tuple[list[str], bool]:
    lower_map = {column.lower(): column for column in frame.columns}
    for token in preferred_tokens:
        if token in lower_map:
            return [lower_map[token]], True

    for token in preferred_tokens:
        token_matches = [
            column
            for column in frame.columns
            if token in column.lower()
        ]
        if token_matches:
            first = token_matches[0]
            if all(
                _looks_like_vector(value)
                for value in _sample_non_null(frame[first])
            ):
                return [first], True
            numeric_matches = [
                column
                for column in token_matches
                if pd.api.types.is_numeric_dtype(frame[column])
            ]
            if numeric_matches:
                return numeric_matches, False

    vector_columns = [
        column
        for column in frame.columns
        if column != id_col
        and all(_looks_like_vector(value) for value in _sample_non_null(frame[column]))
    ]
    if len(vector_columns) == 1:
        return vector_columns, True

    numeric_columns = [
        column
        for column in frame.columns
        if column != id_col
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(numeric_columns) >= 32:
        return numeric_columns, False

    raise ValueError(
        f"Could not infer embedding columns for {id_col}. Available columns: {frame.columns.tolist()}"
    )


def load_embedding_table(
    path: Path,
    *,
    id_col: str,
    preferred_tokens: list[str],
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    columns, is_vector_column = infer_embedding_columns(
        frame,
        id_col=id_col,
        preferred_tokens=preferred_tokens,
    )
    trimmed = frame[[id_col, *columns]].copy()
    trimmed = trimmed.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    trimmed.attrs["embedding_columns"] = columns
    trimmed.attrs["is_vector_column"] = is_vector_column
    return trimmed


def embedding_matrix(frame: pd.DataFrame, *, id_col: str) -> tuple[np.ndarray, np.ndarray]:
    embedding_columns: list[str] = frame.attrs["embedding_columns"]
    is_vector_column: bool = frame.attrs["is_vector_column"]
    ids = frame[id_col].to_numpy()
    if is_vector_column:
        column = embedding_columns[0]
        first_non_null = next(
            value for value in frame[column].tolist() if value is not None
        )
        default_vector = np.zeros_like(np.asarray(first_non_null), dtype=np.float32)
        matrix = np.stack(
            frame[column].map(
                lambda value: default_vector
                if value is None
                else np.asarray(value, dtype=np.float32)
            )
        ).astype(np.float32)
    else:
        matrix = (
            frame[embedding_columns]
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=True)
        )
    return ids, matrix
