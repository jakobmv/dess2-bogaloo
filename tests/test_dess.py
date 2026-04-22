from __future__ import annotations

import unittest

import torch

from dess2_bogaloo.dess_model import (
    DualHeadDessAdapter,
    DualHeadQueryConcatSigmaAdapter,
    FrozenMuSigmaAdapter,
    TextDessAdapter,
)
from dess2_bogaloo.dess_original import DESSLoss as OriginalDESSLoss
from dess2_bogaloo.dess_updated import dess_loss_from_parts, gaussian_log_score


class DessTests(unittest.TestCase):
    def test_original_single_target_loss_runs(self) -> None:
        pred = torch.randn(4, 8)
        target = torch.randn(4, 4)
        loss, mu_loss, sigma_loss = OriginalDESSLoss()(pred, target)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(mu_loss.ndim, 0)
        self.assertEqual(sigma_loss.ndim, 0)

    def test_loss_from_parts_runs_for_detached_sigma_reference(self) -> None:
        mu = torch.randn(3, 4, requires_grad=True)
        sigma = torch.rand(3, 4, requires_grad=True)
        target = torch.randn(3, 4)
        combined, mu_loss, sigma_loss = dess_loss_from_parts(
            mu,
            sigma,
            target,
            reduction="none",
            mu_for_sigma=mu.detach(),
        )
        self.assertEqual(tuple(combined.shape), (3, 8))
        self.assertEqual(tuple(mu_loss.shape), (3, 4))
        self.assertEqual(tuple(sigma_loss.shape), (3, 4))

    def test_joint_adapter_output_shape(self) -> None:
        model = TextDessAdapter(input_dim=4)
        outputs = model(torch.randn(5, 4))
        self.assertEqual(tuple(outputs.mu.shape), (5, 4))
        self.assertEqual(tuple(outputs.sigma.shape), (5, 4))
        self.assertEqual(tuple(outputs.pred.shape), (5, 8))

    def test_frozen_mu_adapter_uses_input_as_mu(self) -> None:
        model = FrozenMuSigmaAdapter(input_dim=4)
        x = torch.randn(5, 4)
        outputs = model(x)
        self.assertTrue(torch.equal(outputs.mu, x))
        self.assertEqual(tuple(outputs.sigma.shape), (5, 4))

    def test_dual_head_detaches_mu_for_sigma(self) -> None:
        model = DualHeadDessAdapter(input_dim=4)
        outputs = model(torch.randn(5, 4))
        self.assertEqual(tuple(outputs.mu.shape), (5, 4))
        self.assertEqual(tuple(outputs.mu_for_sigma.shape), (5, 4))
        self.assertFalse(outputs.mu_for_sigma.requires_grad)

    def test_dual_head_query_concat_sigma_runs(self) -> None:
        model = DualHeadQueryConcatSigmaAdapter(input_dim=4)
        outputs = model(torch.randn(5, 4))
        self.assertEqual(tuple(outputs.mu.shape), (5, 4))
        self.assertEqual(tuple(outputs.sigma.shape), (5, 4))
        scores = gaussian_log_score(outputs.mu, outputs.sigma, torch.randn(5, 4))
        self.assertEqual(tuple(scores.shape), (5,))


if __name__ == "__main__":
    unittest.main()
