from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dess2_bogaloo.data import DatasetPaths, build_reranking_subset


class DataTests(unittest.TestCase):
    def test_build_reranking_subset_filters_task1_test_us(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = DatasetPaths(root)
            paths.esci_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "query_id": 1,
                        "query": "shirt",
                        "product_id": "p1",
                        "product_locale": "us",
                        "esci_label": "E",
                        "small_version": 1,
                        "large_version": 1,
                        "split": "test",
                    },
                    {
                        "query_id": 2,
                        "query": "shirt",
                        "product_id": "p2",
                        "product_locale": "us",
                        "esci_label": "S",
                        "small_version": 1,
                        "large_version": 1,
                        "split": "train",
                    },
                    {
                        "query_id": 3,
                        "query": "shirt",
                        "product_id": "p3",
                        "product_locale": "jp",
                        "esci_label": "E",
                        "small_version": 1,
                        "large_version": 1,
                        "split": "test",
                    },
                ]
            ).to_parquet(paths.examples_path)

            pd.DataFrame(
                [
                    {"product_id": "p1", "product_locale": "us", "product_title": "one"},
                    {"product_id": "p2", "product_locale": "us", "product_title": "two"},
                    {"product_id": "p3", "product_locale": "jp", "product_title": "three"},
                ]
            ).to_parquet(paths.products_path)

            subset = build_reranking_subset(paths)
            self.assertEqual(subset["product_id"].tolist(), ["p1"])
            self.assertEqual(subset["gain"].tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
