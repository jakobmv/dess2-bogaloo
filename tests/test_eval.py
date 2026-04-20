from __future__ import annotations

import math
import unittest

import pandas as pd

from dess2_bogaloo.eval import LABEL_GAINS, evaluate_run, ndcg_for_query


class EvalTests(unittest.TestCase):
    def test_label_gains_match_corrected_paper_mapping(self) -> None:
        self.assertEqual(LABEL_GAINS, {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0})

    def test_ndcg_for_query_is_one_for_ideal_order(self) -> None:
        self.assertTrue(math.isclose(ndcg_for_query([1.0, 0.1, 0.01, 0.0]), 1.0))

    def test_evaluate_run_averages_query_ndcg(self) -> None:
        frame = pd.DataFrame(
            [
                {"query_id": 1, "product_id": "a", "esci_label": "E", "score": 2.0},
                {"query_id": 1, "product_id": "b", "esci_label": "S", "score": 1.0},
                {"query_id": 2, "product_id": "c", "esci_label": "I", "score": 2.0},
                {"query_id": 2, "product_id": "d", "esci_label": "E", "score": 1.0},
            ]
        )
        metrics = evaluate_run(frame)
        self.assertAlmostEqual(metrics["ndcg"], 0.8154648768, places=6)


if __name__ == "__main__":
    unittest.main()
