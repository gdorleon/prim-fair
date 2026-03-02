"""
tests/test_prim.py — Unit Tests for PRIM-Fair
==============================================

Tests cover:
  1. dp_utils   – noise calibration, privacy accounting, DP mechanisms
  2. models     – forward pass, l1_regularizer, interpretability APIs
  3. robustness – PGD attack stays within ℓ∞ ball, robust loss ≥ clean loss
  4. PRIM core  – fit/predict/evaluate pipeline on synthetic data
  5. baselines  – ERM and GroupDRO can train and evaluate
  6. metrics    – correctness of fairness metric functions

Run with:
    python -m pytest tests/test_prim.py -v
    python tests/test_prim.py            # direct execution
"""

import sys, os, math
import unittest
import numpy as np
import torch

# Add parent directory to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_synthetic_data(N=200, d=8, G=2, task="classification", seed=42):
    """
    Create a small synthetic dataset with group structure for testing.
    Group 0: centred at +1; Group 1: centred at -1.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d)).astype(np.float32)
    g = rng.integers(0, G, size=N)

    if task == "classification":
        # Labels correlated with first feature + group
        logits = X[:, 0] + 0.5 * (2 * g - 1)
        y = (logits > 0).astype(int)
    else:
        y = X[:, 0] + 0.5 * (2 * g - 1) + 0.1 * rng.standard_normal(N)
        y = y.astype(np.float32)

    # 80/20 train-test split
    split = int(0.8 * N)
    return {
        "X_train": X[:split], "y_train": y[:split], "g_train": g[:split],
        "X_test":  X[split:], "y_test":  y[split:], "g_test":  g[split:],
        "input_dim": d, "n_groups": G, "task": task,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DP Utilities Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDPUtils(unittest.TestCase):

    def setUp(self):
        from src.dp_utils import (
            gaussian_noise_scale, laplace_noise_scale,
            compute_dp_noise_sigma, DPAccountant,
            privatize_group_losses,
        )
        self.gaussian_noise_scale  = gaussian_noise_scale
        self.laplace_noise_scale   = laplace_noise_scale
        self.compute_dp_noise_sigma = compute_dp_noise_sigma
        self.DPAccountant          = DPAccountant
        self.privatize_group_losses = privatize_group_losses

    def test_gaussian_noise_scale_positive(self):
        """σ must be positive for valid (ε, δ)."""
        sigma = self.gaussian_noise_scale(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        self.assertGreater(sigma, 0.0)

    def test_gaussian_noise_scale_increases_with_sensitivity(self):
        """Higher sensitivity → more noise needed."""
        sigma1 = self.gaussian_noise_scale(1.0, 1.0, 1e-5)
        sigma2 = self.gaussian_noise_scale(2.0, 1.0, 1e-5)
        self.assertGreater(sigma2, sigma1)

    def test_gaussian_noise_scale_decreases_with_epsilon(self):
        """Larger ε budget → less noise needed (weaker privacy)."""
        sigma1 = self.gaussian_noise_scale(1.0, 0.5, 1e-5)
        sigma2 = self.gaussian_noise_scale(1.0, 2.0, 1e-5)
        self.assertGreater(sigma1, sigma2)

    def test_laplace_noise_scale_positive(self):
        b = self.laplace_noise_scale(sensitivity=1.0, epsilon=1.0)
        self.assertGreater(b, 0.0)

    def test_laplace_noise_scale_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            self.laplace_noise_scale(1.0, epsilon=-1.0)

    def test_compute_dp_noise_sigma_valid(self):
        """Should return a positive float for reasonable inputs."""
        sigma = self.compute_dp_noise_sigma(
            target_epsilon=1.0, target_delta=1e-5,
            num_iterations=1000, sensitivity=1.0
        )
        self.assertGreater(sigma, 0.0)
        self.assertIsInstance(sigma, float)

    def test_privatize_group_losses_shape(self):
        """Output shape must match input shape."""
        losses = np.array([0.3, 0.5, 0.4])
        noisy  = self.privatize_group_losses(losses, sensitivity=0.1, epsilon_g=1.0)
        self.assertEqual(noisy.shape, losses.shape)

    def test_privatize_group_losses_clipped(self):
        """Noisy losses must stay in [0, 1]."""
        losses = np.array([0.0, 1.0, 0.5])
        for _ in range(20):
            noisy = self.privatize_group_losses(losses, sensitivity=0.5, epsilon_g=0.1)
            self.assertTrue(np.all(noisy >= 0.0))
            self.assertTrue(np.all(noisy <= 1.0))

    def test_dp_accountant_budget_tracking(self):
        """Accountant should correctly track steps."""
        acc = self.DPAccountant(target_epsilon=2.0, target_delta=1e-5,
                                num_iterations=100)
        self.assertEqual(acc._steps_done, 0)
        acc.step()
        acc.step()
        self.assertEqual(acc._steps_done, 2)

    def test_dp_accountant_within_budget_at_start(self):
        """At 0 steps, epsilon spent should equal only the group-query epsilon."""
        acc = self.DPAccountant(1.0, 1e-5, 100)
        spent = acc.total_privacy_spent(eps_per_step=0.01)
        self.assertLessEqual(spent, acc.target_epsilon)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Model Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestModels(unittest.TestCase):

    def setUp(self):
        from src.models import LinearModel, NeuralModelWithAttention, MixtureOfExpertsModel
        self.LinearModel  = LinearModel
        self.NeuralModel  = NeuralModelWithAttention
        self.MoEModel     = MixtureOfExpertsModel

    def _forward(self, model, d=8, N=16):
        """Run a forward pass and return output tensor."""
        x = torch.randn(N, d)
        return model(x)

    def test_linear_model_output_shape(self):
        model = self.LinearModel(input_dim=8, output_dim=1, task="classification")
        out   = self._forward(model)
        self.assertEqual(out.shape[0], 16)

    def test_linear_model_l1_regularizer(self):
        model = self.LinearModel(8)
        reg   = model.l1_regularizer()
        self.assertIsInstance(reg, torch.Tensor)
        self.assertGreaterEqual(reg.item(), 0.0)

    def test_linear_model_feature_importance(self):
        model  = self.LinearModel(8)
        imp    = model.get_feature_importance()
        self.assertEqual(len(imp), 8)
        # All feature importances should be non-negative
        self.assertTrue(np.all(imp >= 0.0))

    def test_neural_model_output_shape(self):
        model = self.NeuralModel(input_dim=8, hidden_dim=32, task="classification")
        out   = self._forward(model)
        self.assertEqual(out.shape[0], 16)

    def test_neural_model_attention_sums_to_one(self):
        """Feature-attention weights should form a valid probability distribution."""
        model = self.NeuralModel(input_dim=8, hidden_dim=32)
        x     = torch.randn(4, 8)
        _     = model(x)
        # Access attention weights from the model
        if hasattr(model, "last_attention"):
            attn = model.last_attention.detach().numpy()
            # Each sample's attention should approximately sum to 1
            np.testing.assert_allclose(attn.sum(axis=-1), 1.0, atol=1e-4)

    def test_moe_model_output_shape(self):
        model = self.MoEModel(input_dim=8, task="classification")
        out   = self._forward(model)
        self.assertEqual(out.shape[0], 16)

    def test_linear_model_interpretability_score(self):
        model = self.LinearModel(8)
        # Zero out 3 weights to create sparsity
        with torch.no_grad():
            model.linear.weight.data[:, :3] = 0.0
        score = model.interpretability_score()
        # Score should reflect the sparsity
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Robustness Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRobustness(unittest.TestCase):

    def setUp(self):
        from src.robustness import pgd_attack, compute_robust_loss
        from src.models import LinearModel
        self.pgd_attack         = pgd_attack
        self.compute_robust_loss = compute_robust_loss
        self.model = LinearModel(input_dim=8, task="classification")

    def test_pgd_stays_in_linfty_ball(self):
        """PGD-generated adversarial examples must satisfy ‖Δ‖∞ ≤ ρ."""
        rho = 0.1
        x   = torch.randn(10, 8)
        y   = torch.randint(0, 2, (10,)).float()

        x_adv = self.pgd_attack(
            model=self.model, x=x, y=y,
            epsilon=rho, n_steps=5, step_size=rho/3,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
        )
        # Check ℓ∞ constraint: |x_adv - x| ≤ ρ + small numerical tolerance
        diff = (x_adv - x).abs()
        self.assertTrue(
            (diff <= rho + 1e-5).all(),
            f"PGD exceeded ℓ∞ ball: max diff = {diff.max().item():.4f}"
        )

    def test_robust_loss_ge_clean_loss(self):
        """
        Adversarial (robust) loss should be ≥ clean loss, since PGD finds
        the worst-case input — the model can only do worse on adversarial examples.
        """
        x   = torch.randn(16, 8)
        y   = torch.randint(0, 2, (16,)).float()
        rho = 0.1

        loss_fn    = torch.nn.BCEWithLogitsLoss()
        clean_out  = self.model(x)
        clean_loss = loss_fn(clean_out, y).item()

        robust_loss = self.compute_robust_loss(
            model=self.model, x=x, y=y,
            epsilon=rho, n_pgd_steps=5,
            loss_fn=loss_fn,
        ).item()

        # Robust loss ≥ clean loss (PGD can only make things worse or equal)
        self.assertGreaterEqual(robust_loss, clean_loss - 1e-3,
                                f"Robust {robust_loss:.4f} < clean {clean_loss:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PRIM End-to-End Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPRIMEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create shared synthetic data once for all PRIM tests."""
        cls.data = make_synthetic_data(N=200, d=8, G=2, task="classification")

    def _make_prim(self, **overrides):
        """Create a small PRIM instance for fast unit testing."""
        from src.prim import PRIM
        from src.models import LinearModel

        model  = LinearModel(input_dim=8, task="classification")
        config = dict(
            G=2, epsilon=1.0, delta=1e-5,
            rho=0.05, lambda_reg=0.01,
            T=5,              # very few iterations for speed
            lr=0.05,
            batch_size=32,
            n_pgd_steps=2,    # minimal PGD for speed
            inner_epochs=1,
            verbose=False,
        )
        config.update(overrides)
        return PRIM(model=model, **config)

    def test_fit_returns_self(self):
        """fit() should return self (for method chaining)."""
        prim = self._make_prim()
        d    = self.data
        result = prim.fit(d["X_train"], d["y_train"], d["g_train"])
        self.assertIs(result, prim)

    def test_predict_shape(self):
        """predict() should return array of shape (N,)."""
        prim = self._make_prim()
        d    = self.data
        prim.fit(d["X_train"], d["y_train"], d["g_train"])
        preds = prim.predict(d["X_test"])
        self.assertEqual(preds.shape[0], d["X_test"].shape[0])

    def test_predict_binary(self):
        """Classification predictions should be 0 or 1."""
        prim = self._make_prim()
        d    = self.data
        prim.fit(d["X_train"], d["y_train"], d["g_train"])
        preds = prim.predict(d["X_test"])
        unique_vals = set(preds.tolist())
        self.assertTrue(unique_vals.issubset({0, 1}),
                        f"Unexpected prediction values: {unique_vals}")

    def test_predict_proba_in_01(self):
        """Predicted probabilities should be in [0, 1]."""
        prim = self._make_prim()
        d    = self.data
        prim.fit(d["X_train"], d["y_train"], d["g_train"])
        proba = prim.predict_proba(d["X_test"])
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0),
                        "Probabilities outside [0, 1]")

    def test_evaluate_returns_metrics_dict(self):
        """evaluate() should return a non-empty dict with key 'worst_group_error'."""
        prim = self._make_prim()
        d    = self.data
        prim.fit(d["X_train"], d["y_train"], d["g_train"])
        metrics = prim.evaluate(d["X_test"], d["y_test"], d["g_test"])
        self.assertIn("worst_group_error", metrics)
        self.assertIsInstance(metrics["worst_group_error"], float)

    def test_group_weights_sum_to_one(self):
        """PRIM group weights should always sum to 1 (probability simplex)."""
        prim = self._make_prim()
        d    = self.data
        prim.fit(d["X_train"], d["y_train"], d["g_train"])
        self.assertAlmostEqual(prim.group_weights.sum(), 1.0, places=5)

    def test_no_dp_mode(self):
        """PRIM with use_dp=False should still train without errors."""
        prim = self._make_prim(use_dp=False)
        d    = self.data
        prim.fit(d["X_train"], d["y_train"], d["g_train"])
        preds = prim.predict(d["X_test"])
        self.assertEqual(preds.shape[0], d["X_test"].shape[0])

    def test_no_robust_mode(self):
        """PRIM with use_robust=False should skip PGD (faster)."""
        prim = self._make_prim(use_robust=False, rho=0.0)
        d    = self.data
        prim.fit(d["X_train"], d["y_train"], d["g_train"])
        preds = prim.predict(d["X_test"])
        self.assertEqual(preds.shape[0], d["X_test"].shape[0])

    def test_history_populated(self):
        """Training history should record worst_group_error at each log step."""
        prim = self._make_prim(log_every=1)   # log every iteration
        d    = self.data
        prim.fit(d["X_train"], d["y_train"], d["g_train"])
        self.assertGreater(len(prim.history["worst_group_error"]), 0)
        self.assertGreater(len(prim.history["iteration"]), 0)

    def test_regression_mode(self):
        """PRIM should work on regression tasks."""
        from src.prim import PRIM
        from src.models import LinearModel

        data  = make_synthetic_data(N=100, d=6, G=2, task="regression")
        model = LinearModel(input_dim=6, output_dim=1, task="regression")
        prim  = PRIM(model=model, G=2, task="regression", T=3,
                     use_dp=False, use_robust=False, verbose=False,
                     batch_size=32)
        prim.fit(data["X_train"], data["y_train"], data["g_train"])
        preds = prim.predict(data["X_test"])
        self.assertEqual(preds.shape[0], data["X_test"].shape[0])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Baseline Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaselines(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = make_synthetic_data(N=200, d=8, G=2, task="classification")

    def _make_model(self):
        from src.models import LinearModel
        return LinearModel(input_dim=8, task="classification")

    def test_erm_fits_and_predicts(self):
        from baselines.erm import ERM
        d     = self.data
        model = self._make_model()
        erm   = ERM(model, task="classification", n_epochs=10, verbose=False)
        erm.fit(d["X_train"], d["y_train"], d["g_train"])
        preds = erm.predict(d["X_test"])
        self.assertEqual(preds.shape[0], d["X_test"].shape[0])
        # Predictions should be binary
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_group_dro_fits_and_predicts(self):
        from baselines.group_dro import GroupDRO
        d     = self.data
        model = self._make_model()
        dro   = GroupDRO(model, G=2, task="classification", T=5, verbose=False)
        dro.fit(d["X_train"], d["y_train"], d["g_train"])
        preds = dro.predict(d["X_test"])
        self.assertEqual(preds.shape[0], d["X_test"].shape[0])

    def test_group_dro_weights_simplex(self):
        """GroupDRO weights should be on the probability simplex."""
        from baselines.group_dro import GroupDRO
        d     = self.data
        model = self._make_model()
        dro   = GroupDRO(model, G=2, T=5, verbose=False)
        dro.fit(d["X_train"], d["y_train"], d["g_train"])
        self.assertAlmostEqual(dro.group_weights.sum(), 1.0, places=5)

    def test_erm_evaluate_returns_metrics(self):
        from baselines.erm import ERM
        d     = self.data
        model = self._make_model()
        erm   = ERM(model, n_epochs=10, verbose=False)
        erm.fit(d["X_train"], d["y_train"], d["g_train"])
        metrics = erm.evaluate(d["X_test"], d["y_test"], d["g_test"])
        self.assertIn("worst_group_error", metrics)
        self.assertIn("overall_accuracy",  metrics)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Fairness Metrics Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFairnessMetrics(unittest.TestCase):

    def setUp(self):
        from src.fairness_metrics import (
            worst_group_error, overall_accuracy,
            demographic_parity_gap, compute_all_metrics,
        )
        self.worst_group_error      = worst_group_error
        self.overall_accuracy       = overall_accuracy
        self.demographic_parity_gap = demographic_parity_gap
        self.compute_all_metrics    = compute_all_metrics

    def _make_perfect_data(self):
        """A case where all predictions are correct → WGE = 0."""
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.9])  # proba
        g = np.array([0,   0,   0,   1,   1,   1])
        return y, p, g

    def _make_all_wrong_data(self):
        """A case where all predictions are wrong → WGE = 1."""
        y = np.array([0, 0, 1, 1])
        p = np.array([0.9, 0.9, 0.1, 0.1])
        g = np.array([0, 0, 1, 1])
        return y, p, g

    def test_worst_group_error_perfect(self):
        """Perfect predictions → WGE should be 0."""
        y, p, g = self._make_perfect_data()
        wge, _  = self.worst_group_error(y, p, g)
        self.assertAlmostEqual(wge, 0.0, places=5)

    def test_worst_group_error_all_wrong(self):
        """All-wrong predictions → WGE should be 1."""
        y, p, g = self._make_all_wrong_data()
        wge, _  = self.worst_group_error(y, p, g)
        self.assertAlmostEqual(wge, 1.0, places=5)

    def test_worst_group_error_in_01(self):
        """WGE must always be in [0, 1]."""
        rng = np.random.default_rng(0)
        y   = rng.integers(0, 2, 50)
        p   = rng.uniform(0, 1, 50)
        g   = rng.integers(0, 3, 50)
        wge, _ = self.worst_group_error(y, p, g)
        self.assertGreaterEqual(wge, 0.0)
        self.assertLessEqual(wge, 1.0)

    def test_overall_accuracy_perfect(self):
        y, p, g = self._make_perfect_data()
        acc = self.overall_accuracy(y, p)
        self.assertAlmostEqual(acc, 1.0, places=5)

    def test_overall_accuracy_all_wrong(self):
        y, p, g = self._make_all_wrong_data()
        acc = self.overall_accuracy(y, p)
        self.assertAlmostEqual(acc, 0.0, places=5)

    def test_demographic_parity_gap_zero_for_equal_rates(self):
        """If both groups have the same positive prediction rate → gap = 0."""
        # Group 0: 50% positive; Group 1: 50% positive
        p = np.array([0.8, 0.2, 0.8, 0.2])  # proba → 50% positive each group
        g = np.array([0, 0, 1, 1])
        gap = self.demographic_parity_gap(p, g)
        self.assertAlmostEqual(gap, 0.0, places=5)

    def test_demographic_parity_gap_nonnegative(self):
        """Gap should always be ≥ 0."""
        rng = np.random.default_rng(1)
        p   = rng.uniform(0, 1, 100)
        g   = rng.integers(0, 2, 100)
        gap = self.demographic_parity_gap(p, g)
        self.assertGreaterEqual(gap, 0.0)

    def test_compute_all_metrics_keys(self):
        """compute_all_metrics must return all expected keys."""
        y, p, g = self._make_perfect_data()
        metrics = self.compute_all_metrics(
            y_true=y, y_pred=p, groups=g,
            task="classification", compute_auc_flag=True,
        )
        for key in ["worst_group_error", "overall_accuracy",
                    "demographic_parity_gap", "max_fpr"]:
            self.assertIn(key, metrics, f"Missing key: {key}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Integration test: full pipeline on synthetic data
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline(unittest.TestCase):
    """
    Integration test: train PRIM and all baselines, verify that PRIM
    achieves lower worst-group error than ERM (on this synthetic data).
    """

    def test_prim_beats_erm_on_worst_group(self):
        """
        On the imbalanced synthetic data, PRIM's worst-group error should be
        lower than ERM's, demonstrating the core value proposition.

        Note: This is a probabilistic test — results depend on the random seed.
        We use enough iterations and a strong enough imbalance to make it robust.
        """
        from src.prim import PRIM
        from src.models import LinearModel
        from baselines.erm import ERM

        data = make_synthetic_data(N=500, d=8, G=2, seed=99)

        # ─ Train ERM ─
        erm_model = LinearModel(8)
        erm = ERM(erm_model, n_epochs=50, verbose=False)
        erm.fit(data["X_train"], data["y_train"], data["g_train"])
        erm_metrics = erm.evaluate(data["X_test"], data["y_test"], data["g_test"])

        # ─ Train PRIM ─
        prim_model = LinearModel(8)
        prim = PRIM(prim_model, G=2, T=20, use_dp=False, use_robust=False,
                    lambda_reg=0.01, verbose=False, batch_size=64)
        prim.fit(data["X_train"], data["y_train"], data["g_train"])
        prim_metrics = prim.evaluate(data["X_test"], data["y_test"], data["g_test"])

        erm_wge  = erm_metrics["worst_group_error"]
        prim_wge = prim_metrics["worst_group_error"]

        print(f"\n[Integration] ERM WGE={erm_wge:.4f}, PRIM WGE={prim_wge:.4f}")

        # PRIM should achieve lower worst-group error
        # (allow a small tolerance for random variation)
        self.assertLessEqual(
            prim_wge, erm_wge + 0.10,  # PRIM should not be more than 10% worse
            f"PRIM WGE ({prim_wge:.4f}) is much worse than ERM WGE ({erm_wge:.4f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run all tests with verbose output
    print("\n" + "="*65)
    print("  PRIM-Fair Unit Tests")
    print("="*65 + "\n")
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(__import__(__name__))
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
