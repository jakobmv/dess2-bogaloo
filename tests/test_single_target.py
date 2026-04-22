from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dess2_bogaloo.single_target import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    FrozenLinearMuSigmaAdapter,
    GasTurbinePaths,
    build_gas_turbine_splits,
)
from dess2_bogaloo.dess_model import (
    DualHeadDessAdapter,
    DualHeadQueryConcatSigmaAdapter,
    TextDessAdapter,
)


class SingleTargetTests(unittest.TestCase):
    def test_single_target_variants_produce_target_dim_outputs(self) -> None:
        x = torch.randn(5, 9)
        target_dim = 2
        train_x = np.random.randn(20, 9).astype(np.float32)
        train_y = np.random.randn(20, target_dim).astype(np.float32)
        models = [
            TextDessAdapter(input_dim=9, output_dim=target_dim, hidden_dim=16),
            FrozenLinearMuSigmaAdapter(
                input_dim=9,
                output_dim=target_dim,
                train_features=train_x,
                train_targets=train_y,
                hidden_dim=16,
            ),
            DualHeadDessAdapter(input_dim=9, output_dim=target_dim, hidden_dim=16),
            DualHeadQueryConcatSigmaAdapter(input_dim=9, output_dim=target_dim, hidden_dim=16),
        ]
        for model in models:
            outputs = model(x)
            self.assertEqual(tuple(outputs.mu.shape), (5, target_dim))
            self.assertEqual(tuple(outputs.sigma.shape), (5, target_dim))

    def test_gas_turbine_split_is_chronological_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = GasTurbinePaths(Path(temp_dir))
            splits = build_gas_turbine_splits(paths, val_fraction=0.2)
            total_rows = sum(frame.shape[0] for frame in splits.values())
            self.assertEqual(total_rows, 36733)
            self.assertEqual(splits["test"]["year"].min(), 2014)
            self.assertEqual(splits["test"]["year"].max(), 2015)
            self.assertEqual(set(FEATURE_COLUMNS).issubset(splits["train"].columns), True)
            self.assertEqual(set(TARGET_COLUMNS).issubset(splits["train"].columns), True)


if __name__ == "__main__":
    unittest.main()
