from __future__ import annotations

import unittest

import numpy as np

from dess2_bogaloo.dess_sampling import sample_candidate_order


class DessSamplingTests(unittest.TestCase):
    def test_sample_candidate_order_returns_a_full_permutation(self) -> None:
        mu = np.array([1.0, 0.0], dtype=np.float32)
        sigma = np.array([0.1, 0.1], dtype=np.float32)
        candidates = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        draw_order, draw_cosine = sample_candidate_order(
            mu=mu,
            sigma=sigma,
            candidate_matrix=candidates,
            rng=np.random.default_rng(42),
        )
        self.assertEqual(sorted(draw_order.tolist()), [1, 2, 3])
        self.assertEqual(tuple(draw_cosine.shape), (3,))

    def test_sampling_is_reproducible_for_same_rng_seed(self) -> None:
        mu = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        sigma = np.full(4, 0.5, dtype=np.float32)
        candidates = np.eye(4, dtype=np.float32)
        first_order, _ = sample_candidate_order(
            mu=mu,
            sigma=sigma,
            candidate_matrix=candidates,
            rng=np.random.default_rng(7),
        )
        second_order, _ = sample_candidate_order(
            mu=mu,
            sigma=sigma,
            candidate_matrix=candidates,
            rng=np.random.default_rng(7),
        )
        self.assertTrue(np.array_equal(first_order, second_order))


if __name__ == "__main__":
    unittest.main()
