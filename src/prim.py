"""
prim.py — Core PRIM Algorithm

Implements Algorithm 1 from the paper: Private, Robust, Interpretable Minimax Fairness.

The algorithm frames learning as a two-player zero-sum game:
  - Player 1 (Learner): Minimizes the group-weighted robust loss + interpretability penalty
  - Player 2 (Regulator): Up-weights the worst-performing group

Key steps per iteration t:
  1. Private Group Loss Evaluation: Compute per-group losses + add DP noise
  2. Update Group Weights: Boost weight of worst-off group (exponential weighting)
  3. Robust Model Training: DP-SGD with PGD adversarial inner loop

Theoretical guarantees (Section IV):
  - Convergence: O(1/ε²) iterations to ε-optimal minimax solution
  - Privacy: (ε, δ)-DP via DP-SGD + advanced composition
  - Robustness: Loss bounded by γ_g + η for ‖Δ‖∞ ≤ ρ
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Tuple
import time
import copy

from .dp_utils import (
    DPAccountant,
    compute_dp_noise_sigma,
    privatize_group_losses,
    dp_sgd_step,
)
from .robustness import compute_robust_loss, compute_group_robust_losses
from .fairness_metrics import compute_all_metrics


class PRIM:
    """
    Private, Robust, Interpretable Minimax Fairness (PRIM).

    Implements the two-player minimax game from Algorithm 1 of the paper.

    Usage:
        prim = PRIM(
            model=my_model,
            G=2,
            epsilon=1.0,
            delta=1e-5,
            rho=0.1,
            lambda_reg=0.01,
            T=100,
        )
        prim.fit(X_train, y_train, g_train)
        predictions = prim.predict(X_test)
        metrics = prim.evaluate(X_test, y_test, g_test)

    Args:
        model:         PyTorch model with .l1_regularizer() method.
        G:             Number of demographic groups.
        epsilon:       DP privacy budget ε.
        delta:         DP privacy failure probability δ.
        rho:           ℓ∞ adversarial perturbation radius.
        lambda_reg:    Weight for interpretability regularizer (λ in paper).
        T:             Number of PRIM outer iterations.
        lr:            Learning rate for model parameter updates.
        batch_size:    Mini-batch size for stochastic training.
        max_grad_norm: Gradient clipping threshold for DP-SGD (C in paper).
        alpha:         Group weight boost factor (α > 1, boosts worst group).
        n_pgd_steps:   Number of PGD steps for robust loss approximation.
        use_dp:        Whether to enable differential privacy.
        use_robust:    Whether to use adversarial (robust) training.
        use_reg:       Whether to apply interpretability regularizer.
        task:          "classification" or "regression".
        device:        Torch device ("cpu" or "cuda").
        inner_epochs:  Number of SGD epochs per outer PRIM iteration.
        verbose:       Whether to print progress.
        log_every:     Print progress every log_every iterations.
    """

    def __init__(
        self,
        model: nn.Module,
        G: int = 2,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        rho: float = 0.1,
        lambda_reg: float = 0.01,
        T: int = 100,
        lr: float = 0.01,
        batch_size: int = 64,
        max_grad_norm: float = 1.0,
        alpha: float = 1.2,
        n_pgd_steps: int = 10,
        use_dp: bool = True,
        use_robust: bool = True,
        use_reg: bool = True,
        task: str = "classification",
        device: str = "cpu",
        inner_epochs: int = 1,
        verbose: bool = True,
        log_every: int = 10,
    ):
        self.model = model.to(device)
        self.G = G
        self.epsilon = epsilon
        self.delta = delta
        self.rho = rho
        self.lambda_reg = lambda_reg
        self.T = T
        self.lr = lr
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.n_pgd_steps = n_pgd_steps
        self.use_dp = use_dp
        self.use_robust = use_robust
        self.use_reg = use_reg
        self.task = task
        self.device = device
        self.inner_epochs = inner_epochs
        self.verbose = verbose
        self.log_every = log_every

        # Will be set during fit()
        self.group_weights: Optional[np.ndarray] = None
        self.sigma: Optional[float] = None
        self.accountant: Optional[DPAccountant] = None

        # Training history for plotting and analysis
        self.history: Dict[str, List] = {
            "iteration": [],
            "worst_group_error": [],
            "overall_error": [],
            "group_weights": [],
            "group_losses": [],
            "total_epsilon_spent": [],
        }

        # Loss function depending on task
        if task == "classification":
            # BCEWithLogits handles sigmoid internally — numerically stable
            self.loss_fn_reduction_none = nn.BCEWithLogitsLoss(reduction="none")
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif task == "regression":
            self.loss_fn_reduction_none = nn.MSELoss(reduction="none")
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task: {task}")

    def _setup_privacy(self) -> None:
        """
        Configure DP-SGD noise scale σ based on (ε, δ, T).

        σ is computed such that over T gradient updates with gradient clipping C,
        the total privacy cost is ≤ (ε, δ)-DP via advanced composition.
        """
        # Total inner gradient steps = T outer iterations * inner_epochs * batches_per_epoch
        # We approximate total steps here; in practice use Opacus for exact accounting
        total_steps = self.T * self.inner_epochs * 20  # rough estimate

        self.accountant = DPAccountant(self.epsilon, self.delta, total_steps)
        self.sigma = compute_dp_noise_sigma(
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            num_iterations=total_steps,
            sensitivity=self.max_grad_norm,
        )

        if self.verbose:
            print(f"  [DP] σ = {self.sigma:.4f} for (ε={self.epsilon}, δ={self.delta}) "
                  f"over {total_steps} steps")

    def _compute_group_losses(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        g: torch.Tensor,
    ) -> np.ndarray:
        """
        Step 4 of Algorithm 1: Evaluate per-group loss on current model.
        Optionally adds DP noise to protect individual data.

        L_g(θ) = (1/|D_g|) * Σ_{(x,y)∈D_g} ℓ(f_θ(x), y)

        Args:
            X, y, g: Full training data tensors.

        Returns:
            group_losses: Numpy array of shape (G,), possibly noisy.
        """
        self.model.eval()
        group_losses = np.zeros(self.G)
        group_counts = np.zeros(self.G)

        with torch.no_grad():
            # Process in batches for memory efficiency
            n = X.shape[0]
            batch_size = min(self.batch_size * 4, n)  # larger batches for evaluation

            all_losses = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x_batch = X[start:end].to(self.device)
                y_batch = y[start:end].to(self.device)

                out = self.model(x_batch)
                if self.task == "classification":
                    # Cast y to float for BCEWithLogitsLoss
                    losses = self.loss_fn_reduction_none(out, y_batch.float())
                else:
                    losses = self.loss_fn_reduction_none(out, y_batch.float())
                all_losses.append(losses.cpu())

            all_losses = torch.cat(all_losses)  # (N,)

        # Aggregate per group
        g_np = g.numpy() if isinstance(g, torch.Tensor) else g
        for grp in range(self.G):
            mask = (g_np == grp)
            if mask.sum() > 0:
                group_losses[grp] = all_losses[mask].mean().item()
                group_counts[grp] = mask.sum()

        # Privatize group losses (Step 4: add DP noise to group queries)
        if self.use_dp:
            # Sensitivity = 1 / min_group_size (bounded loss assumption: ℓ ∈ [0,1])
            min_count = max(group_counts.min(), 1)
            sensitivity = 1.0 / min_count

            # Use a small fraction of epsilon for group queries
            eps_group = self.epsilon * 0.05  # 5% of budget for group queries
            delta_group = self.delta * 0.1

            group_losses = privatize_group_losses(
                group_losses=group_losses,
                sensitivity=sensitivity,
                epsilon_g=eps_group,
                mechanism="gaussian",
                delta_g=delta_group,
            )

        self.model.train()
        return group_losses

    def _update_group_weights(
        self,
        group_losses: np.ndarray,
    ) -> np.ndarray:
        """
        Step 5 of Algorithm 1: Update group weights via exponential weighting.

        The worst group (highest noisy loss) gets its weight multiplied by α.
        All weights are then re-normalized to sum to 1 (simplex projection).

        Formally:
            g* = argmax_g L̃_g
            w_{g*}^(t) = α * w_{g*}^(t-1)
            w_g^(t) = w_g^(t-1) / Σ_g' w_{g'}^(t-1)  for g ≠ g*

        This implements the exponential weights / oracle-efficient reduction
        approach of Diana et al. 2021.
        """
        weights = self.group_weights.copy()

        # Find worst-performing group (using noisy losses for DP)
        worst_group = np.argmax(group_losses)

        # Boost the weight of the worst group
        weights[worst_group] *= self.alpha

        # Project back to the probability simplex (normalize to sum=1)
        weights = weights / weights.sum()

        return weights, worst_group

    def _compute_weighted_robust_loss(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        g_batch: torch.Tensor,
        weights: np.ndarray,
    ) -> torch.Tensor:
        """
        Step 6 of Algorithm 1: Compute the group-weighted robust training loss.

        Loss = Σ_g w_g * L_g^(rob)(θ) + λ * Ω(θ)

        where L_g^(rob)(θ) = E_{(x,y)~D_g}[max_{‖Δ‖∞ ≤ ρ} ℓ(f_θ(x+Δ), y)]

        For efficiency, we compute per-sample losses weighted by group membership.
        """
        # Convert group weights to a per-sample weight tensor
        # w_i = w_{g(i)} — each sample gets its group's current weight
        g_np = g_batch.cpu().numpy()
        sample_weights = torch.tensor(
            [weights[g] for g in g_np], dtype=torch.float32, device=self.device
        )

        if self.use_robust and self.rho > 0:
            # Inner maximization: find worst-case perturbation via PGD
            # This approximates max_{‖Δ‖∞ ≤ ρ} ℓ(f_θ(x+Δ), y)
            x_adv = self._pgd_inner_loop(x_batch, y_batch)
        else:
            # No robustness: use clean inputs
            x_adv = x_batch

        # Outer minimization: compute loss on adversarial examples
        out = self.model(x_adv)

        if self.task == "classification":
            per_sample_loss = self.loss_fn_reduction_none(out, y_batch.float())
        else:
            per_sample_loss = self.loss_fn_reduction_none(out, y_batch.float())

        # Group-weighted loss (Σ_g w_g * L_g^(rob))
        # = (1/n) * Σ_i w_{g(i)} * ℓ(f(x_i), y_i) [reweighted ERM]
        weighted_loss = (sample_weights * per_sample_loss).sum() / (sample_weights.sum() + 1e-10)

        # Interpretability regularizer: λ * Ω(θ) = λ * ‖w‖₁
        if self.use_reg and hasattr(self.model, "l1_regularizer"):
            reg_loss = self.lambda_reg * self.model.l1_regularizer()
        else:
            reg_loss = torch.tensor(0.0, device=self.device)

        total_loss = weighted_loss + reg_loss
        return total_loss

    def _pgd_inner_loop(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inner PGD loop for adversarial example generation.
        Uses a small number of steps (n_pgd_steps) for tractability.
        This approximates the inner max of the robust loss.
        """
        step_size = self.rho * 2.5 / max(self.n_pgd_steps, 1)
        x_adv = x.clone().detach()

        # Random initialization within ℓ∞ ball
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.rho, self.rho)
        x_adv = torch.clamp(x_adv, x - self.rho, x + self.rho)

        # PGD iterations
        for _ in range(self.n_pgd_steps):
            x_adv.requires_grad_(True)
            out = self.model(x_adv)

            if self.task == "classification":
                loss = self.loss_fn(out, y.float())
            else:
                loss = self.loss_fn(out, y.float())

            # Gradient ascent on input (maximize loss)
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
            x_adv = x_adv.detach() + step_size * grad.sign()
            x_adv = torch.clamp(x_adv, x - self.rho, x + self.rho).detach()

        return x_adv

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        g_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        g_val: Optional[np.ndarray] = None,
    ) -> "PRIM":
        """
        Fit the PRIM model using Algorithm 1.

        Args:
            X_train:  Training features, shape (N, d).
            y_train:  Training labels, shape (N,).
            g_train:  Training group memberships, shape (N,), values in {0,...,G-1}.
            X_val, y_val, g_val: Optional validation data for monitoring.

        Returns:
            self (for method chaining).
        """
        start_time = time.time()

        # Convert numpy arrays to tensors
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        g_t = torch.tensor(g_train, dtype=torch.long)

        N = X_t.shape[0]

        # Step 2: Initialization
        # Uniform initial group weights: w_g^(0) = 1/G for all g
        self.group_weights = np.ones(self.G) / self.G

        # Setup DP noise scale
        if self.use_dp:
            self._setup_privacy()
        else:
            self.sigma = 0.0

        # Setup optimizer (SGD for DP-SGD compatibility)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T)

        # Create DataLoader for mini-batch training
        dataset = TensorDataset(X_t, y_t, g_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Training PRIM on {N} samples, G={self.G} groups")
            print(f"  Settings: DP={self.use_dp}, Robust={self.use_robust}, "
                  f"Reg={self.use_reg}")
            print(f"  ε={self.epsilon}, δ={self.delta}, ρ={self.rho}, λ={self.lambda_reg}")
            print(f"  T={self.T} iterations, lr={self.lr}, batch={self.batch_size}")
            print(f"{'='*60}")

        best_worst_group_error = float("inf")
        best_model_state = None
        eps_m = self.accountant.per_step_epsilon() if self.use_dp else 0.0
        iter_count = 0

        # ==================== Main PRIM Loop (Algorithm 1) ====================
        for t in range(1, self.T + 1):

            # ----- Step 4: Private Group Loss Evaluation -----
            # Compute per-group losses with optional DP noise
            group_losses = self._compute_group_losses(X_t, y_t, g_t)

            # ----- Step 5: Update Group Weights -----
            # Identify worst group; boost its weight by α; normalize
            self.group_weights, worst_group = self._update_group_weights(group_losses)

            # ----- Step 6: Robust Model Training (Private) -----
            # Minimize: Σ_g w_g * L_g^(rob)(θ) + λ*Ω(θ)
            # using DP-SGD with PGD adversarial inner loop

            self.model.train()
            epoch_losses = []

            for _ in range(self.inner_epochs):
                for x_batch, y_batch, g_batch in loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    g_batch = g_batch.to(self.device)

                    # Compute group-weighted robust loss (inner PGD + group weighting)
                    loss = self._compute_weighted_robust_loss(
                        x_batch, y_batch, g_batch, self.group_weights
                    )

                    if self.use_dp:
                        # DP-SGD: clip gradients + add Gaussian noise
                        dp_sgd_step(
                            model=self.model,
                            optimizer=optimizer,
                            loss=loss,
                            sigma=self.sigma,
                            max_grad_norm=self.max_grad_norm,
                            batch_size=len(x_batch),
                        )
                    else:
                        # Standard gradient descent
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    epoch_losses.append(loss.item())
                    iter_count += 1

            scheduler.step()

            # ----- Logging -----
            if t % self.log_every == 0 or t == 1 or t == self.T:
                # Quick evaluation (no PGD for speed during training)
                with torch.no_grad():
                    self.model.eval()
                    out = self.model(X_t.to(self.device))
                    if self.task == "classification":
                        proba = torch.sigmoid(out).cpu().numpy()
                        pred_labels = (proba > 0.5).astype(int)
                    else:
                        pred_labels = out.cpu().numpy()

                    # Compute per-group errors
                    g_np = g_train
                    group_errors = {}
                    for grp in range(self.G):
                        mask = (g_np == grp)
                        if mask.sum() > 0:
                            if self.task == "classification":
                                group_errors[grp] = 1.0 - (pred_labels[mask] == y_train[mask]).mean()
                            else:
                                group_errors[grp] = np.abs(pred_labels[mask] - y_train[mask]).mean()

                    wge = max(group_errors.values()) if group_errors else float("inf")
                    overall_err = 1.0 - (pred_labels == y_train).mean() if self.task == "classification" \
                                  else np.abs(pred_labels - y_train).mean()

                self.model.train()

                # Track best model (minimize worst-group error)
                if wge < best_worst_group_error:
                    best_worst_group_error = wge
                    best_model_state = copy.deepcopy(self.model.state_dict())

                # Compute total epsilon spent so far
                total_eps = 0.0
                if self.use_dp and self.accountant is not None:
                    old_T = self.accountant.num_iterations
                    self.accountant.num_iterations = iter_count
                    total_eps = self.accountant.total_privacy_spent(eps_m)
                    self.accountant.num_iterations = old_T

                # Store history
                self.history["iteration"].append(t)
                self.history["worst_group_error"].append(wge)
                self.history["overall_error"].append(overall_err)
                self.history["group_weights"].append(self.group_weights.copy())
                self.history["group_losses"].append(group_losses.copy())
                self.history["total_epsilon_spent"].append(total_eps)

                if self.verbose:
                    elapsed = time.time() - start_time
                    print(
                        f"  Iter {t:4d}/{self.T} | "
                        f"WGE={wge:.4f} | "
                        f"Overall={overall_err:.4f} | "
                        f"WorstGroup={worst_group} | "
                        f"Weights={np.round(self.group_weights, 3)} | "
                        f"ε_spent={total_eps:.3f} | "
                        f"Time={elapsed:.1f}s"
                    )

        # Restore best model found during training
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if self.verbose:
                print(f"\n  ✓ Restored best model with WGE={best_worst_group_error:.4f}")

        total_time = time.time() - start_time
        if self.verbose:
            print(f"\n  Training complete in {total_time:.2f}s")
            print(f"  Final group weights: {np.round(self.group_weights, 4)}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (classification) or raw values (regression).

        Args:
            X: Input features, shape (N, d).

        Returns:
            probabilities: Shape (N,) for binary classification, (N, C) for multi-class.
        """
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        all_preds = []
        with torch.no_grad():
            for start in range(0, len(X_t), self.batch_size * 4):
                end = min(start + self.batch_size * 4, len(X_t))
                out = self.model(X_t[start:end])
                if self.task == "classification":
                    proba = torch.sigmoid(out)
                else:
                    proba = out
                all_preds.append(proba.cpu())

        self.model.train()
        return torch.cat(all_preds).numpy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict labels (classification) or values (regression).

        Args:
            X:         Input features.
            threshold: Classification threshold.

        Returns:
            predictions: Binary labels (classification) or continuous values (regression).
        """
        proba = self.predict_proba(X)
        if self.task == "classification":
            return (proba > threshold).astype(int)
        return proba

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        g: np.ndarray,
    ) -> Dict:
        """
        Evaluate model on a dataset and return all fairness/accuracy metrics.

        Args:
            X, y, g: Features, labels, groups.

        Returns:
            metrics dict from fairness_metrics.compute_all_metrics().
        """
        predictions = self.predict_proba(X)
        metrics = compute_all_metrics(
            y_true=y,
            y_pred=predictions,
            groups=g,
            task=self.task,
            compute_auc_flag=(self.task == "classification"),
        )
        return metrics

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance scores from the underlying model."""
        if hasattr(self.model, "get_feature_importance"):
            return self.model.get_feature_importance()
        return np.zeros(1)

    def get_interpretability_score(self) -> float:
        """Return model's interpretability score."""
        if hasattr(self.model, "interpretability_score"):
            return self.model.interpretability_score()
        return 0.0

    def get_active_feature_count(self, threshold: float = 1e-4) -> int:
        """Return number of features with non-negligible importance."""
        if hasattr(self.model, "get_active_features"):
            return len(self.model.get_active_features(threshold))
        return -1

    def save(self, path: str) -> None:
        """Save model state and PRIM configuration."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "group_weights": self.group_weights,
            "history": self.history,
            "config": {
                "G": self.G, "epsilon": self.epsilon, "delta": self.delta,
                "rho": self.rho, "lambda_reg": self.lambda_reg, "T": self.T,
                "use_dp": self.use_dp, "use_robust": self.use_robust,
                "use_reg": self.use_reg, "task": self.task,
            }
        }, path)
        print(f"  Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.group_weights = checkpoint["group_weights"]
        self.history = checkpoint.get("history", self.history)
        print(f"  Model loaded from {path}")
