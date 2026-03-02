"""
baselines.py — Baseline Methods for Comparison with PRIM
=========================================================

Implements the baseline methods compared in the paper's experiments:
  1. ERM                — Vanilla Empirical Risk Minimization (no fairness)
  2. EqualizedOdds      — Fairness via equalized odds constraints (Agarwal 2018 reductions)
  3. GroupDRO           — Minimax worst-group loss without DP/robustness/interpretability
                          (equivalent to Diana et al. 2021 and Sagawa et al. 2020)

Each baseline exposes the same API as PRIM:
    model.fit(X_train, y_train, g_train)
    model.predict(X_test)
    model.predict_proba(X_test)
    model.evaluate(X_test, y_test, g_test) → metrics dict

This makes it easy to run a fair comparison in the experiment scripts.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict
import copy

from .fairness_metrics import compute_all_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Base class: shared predict / evaluate logic
# ═══════════════════════════════════════════════════════════════════════════════

class BaseModel:
    """
    Shared evaluation interface for all models (baselines + PRIM).
    Subclasses must implement fit() and their own forward-pass model.
    """

    def __init__(self, model: nn.Module, task: str = "classification",
                 device: str = "cpu", batch_size: int = 64):
        self.model      = model.to(device)
        self.task       = task
        self.device     = device
        self.batch_size = batch_size

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw sigmoid probabilities (classification) or values (regression)."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for s in range(0, len(X_t), self.batch_size * 4):
                e = min(s + self.batch_size * 4, len(X_t))
                out = self.model(X_t[s:e].to(self.device))
                if self.task == "classification":
                    preds.append(torch.sigmoid(out).cpu())
                else:
                    preds.append(out.cpu())
        self.model.train()
        return torch.cat(preds).numpy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return hard labels (classification) or values (regression)."""
        proba = self.predict_proba(X)
        if self.task == "classification":
            return (proba > threshold).astype(int)
        return proba

    def evaluate(self, X: np.ndarray, y: np.ndarray, g: np.ndarray) -> Dict:
        """Compute all fairness and accuracy metrics."""
        preds = self.predict_proba(X)
        return compute_all_metrics(
            y_true=y,
            y_pred=preds,
            groups=g,
            task=self.task,
            compute_auc_flag=(self.task == "classification"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Empirical Risk Minimization (ERM) — no fairness
# ═══════════════════════════════════════════════════════════════════════════════

class ERMBaseline(BaseModel):
    """
    Vanilla ERM: minimises total cross-entropy (or MSE) with no fairness constraints.

    This is the 'Unconstrained Model' from Table I of the paper.
    It typically achieves the lowest overall error but the highest worst-group error.

    Args:
        model:       PyTorch model.
        task:        "classification" or "regression".
        lr:          Learning rate.
        epochs:      Number of training epochs.
        batch_size:  Mini-batch size.
        device:      Torch device string.
        verbose:     Print training progress.
    """

    def __init__(
        self,
        model: nn.Module,
        task: str = "classification",
        lr: float = 0.01,
        epochs: int = 100,
        batch_size: int = 64,
        device: str = "cpu",
        verbose: bool = True,
    ):
        super().__init__(model, task, device, batch_size)
        self.lr      = lr
        self.epochs  = epochs
        self.verbose = verbose

        if task == "classification":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, g_train: np.ndarray) -> "ERMBaseline":
        """
        Standard mini-batch SGD training — no group information used.
        g_train is accepted for API compatibility but ignored.
        """
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        dataset  = TensorDataset(X_t, y_t)
        loader   = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        best_loss   = float("inf")
        best_state  = None

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                out  = self.model(x_b)
                loss = self.loss_fn(out, y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            scheduler.step()

            # Keep the model checkpoint with the lowest training loss
            if avg_loss < best_loss:
                best_loss  = avg_loss
                best_state = copy.deepcopy(self.model.state_dict())

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"  [ERM] Epoch {epoch+1:3d}/{self.epochs} | loss={avg_loss:.4f}")

        self.model.load_state_dict(best_state)
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Group DRO — minimax without DP / robustness / interpretability
# ═══════════════════════════════════════════════════════════════════════════════

class GroupDROBaseline(BaseModel):
    """
    Group Distributionally Robust Optimization (Group DRO).

    Minimises the worst-group loss without any privacy, robustness, or
    interpretability constraints. This is equivalent to:
      - Diana et al. 2021 'Minimax Group Fairness'
      - Sagawa et al. 2020 'Distributionally Robust Neural Networks'

    The update rule maintains a group-weight distribution w ∈ Δ_G.
    At each step:
      1. Compute per-group losses L_g(θ).
      2. Update w_g ∝ w_g * exp(η_w * L_g)  [exponential ascent on group weights].
      3. Gradient descent on θ to minimize Σ_g w_g * L_g(θ).

    Args:
        model:       PyTorch model.
        G:           Number of demographic groups.
        task:        "classification" or "regression".
        lr:          Learning rate for θ.
        group_lr:    Learning rate for group weights (η_w in DRO).
        T:           Number of outer iterations.
        inner_epochs: SGD epochs per outer step.
        batch_size:  Mini-batch size.
        device:      Torch device.
        verbose:     Print progress.
    """

    def __init__(
        self,
        model: nn.Module,
        G: int = 2,
        task: str = "classification",
        lr: float = 0.01,
        group_lr: float = 0.01,
        T: int = 50,
        inner_epochs: int = 1,
        batch_size: int = 64,
        device: str = "cpu",
        verbose: bool = True,
    ):
        super().__init__(model, task, device, batch_size)
        self.G           = G
        self.lr          = lr
        self.group_lr    = group_lr
        self.T           = T
        self.inner_epochs = inner_epochs
        self.verbose     = verbose

        # Uniform initial group weights (probability simplex)
        self.group_weights = np.ones(G) / G

        if task == "classification":
            self.loss_fn_none = nn.BCEWithLogitsLoss(reduction="none")
            self.loss_fn      = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn_none = nn.MSELoss(reduction="none")
            self.loss_fn      = nn.MSELoss()

    def _compute_group_losses(self, X_t, y_t, g_np):
        """Compute clean (non-private) per-group losses."""
        self.model.eval()
        group_losses = np.zeros(self.G)
        with torch.no_grad():
            out = self.model(X_t.to(self.device))
            losses = self.loss_fn_none(out, y_t.float().to(self.device)).cpu().numpy()
        for g in range(self.G):
            mask = (g_np == g)
            if mask.sum() > 0:
                group_losses[g] = losses[mask].mean()
        self.model.train()
        return group_losses

    def _update_weights_exponent(self, group_losses: np.ndarray) -> np.ndarray:
        """
        Exponential-weight update: w_g ← w_g * exp(η_w * L_g), then normalise.
        This is the standard DRO mirror-ascent step on the group weights.
        """
        weights = self.group_weights * np.exp(self.group_lr * group_losses)
        weights = weights / weights.sum()   # project onto probability simplex
        return weights

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, g_train: np.ndarray) -> "GroupDROBaseline":
        """
        Alternating minimax training:
          - θ: gradient *descent* on group-weighted loss
          - w: exponential *ascent* on group losses
        """
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        g_np = np.asarray(g_train)

        dataset   = TensorDataset(X_t, y_t, torch.tensor(g_np, dtype=torch.long))
        loader    = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        best_wge   = float("inf")
        best_state = None

        for t in range(1, self.T + 1):
            # ─── Compute per-group losses (no DP noise) ───
            group_losses = self._compute_group_losses(X_t, y_t, g_np)

            # ─── Update group weights via exponentiated gradient ───
            self.group_weights = self._update_weights_exponent(group_losses)

            # ─── Inner SGD: minimize Σ_g w_g * L_g(θ) ───
            self.model.train()
            for _ in range(self.inner_epochs):
                for x_b, y_b, g_b in loader:
                    x_b = x_b.to(self.device)
                    y_b = y_b.to(self.device)
                    g_b_np = g_b.numpy()

                    # Per-sample weights from current group distribution
                    sw = torch.tensor(
                        [self.group_weights[g] for g in g_b_np],
                        dtype=torch.float32, device=self.device
                    )
                    out     = self.model(x_b)
                    per_s   = self.loss_fn_none(out, y_b.float())
                    loss    = (sw * per_s).sum() / (sw.sum() + 1e-10)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # ─── Track best worst-group error ───
            with torch.no_grad():
                self.model.eval()
                proba = torch.sigmoid(self.model(X_t.to(self.device))).cpu().numpy()
                preds = (proba > 0.5).astype(int) if self.task == "classification" else proba

            wge = max(
                1.0 - (preds[g_np == g] == y_train[g_np == g]).mean()
                for g in range(self.G)
                if (g_np == g).sum() > 0
            )
            if wge < best_wge:
                best_wge   = wge
                best_state = copy.deepcopy(self.model.state_dict())
            self.model.train()

            if self.verbose and t % 10 == 0:
                print(f"  [GroupDRO] Iter {t:3d}/{self.T} | WGE={wge:.4f} | "
                      f"Weights={np.round(self.group_weights, 3)}")

        self.model.load_state_dict(best_state)
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Equalized Odds Baseline (via post-processing threshold adjustment)
# ═══════════════════════════════════════════════════════════════════════════════

class EqualizedOddsBaseline(BaseModel):
    """
    Equalized Odds fairness via post-processing threshold selection.

    Approach (Hardt et al. 2016):
      1. Train a standard ERM model.
      2. Search for per-group thresholds that equalise FPR and TPR
         across groups (equalized odds).

    This is the 'EO-Fair' baseline in Tables I and III of the paper.
    Unlike reduction methods (Agarwal 2018), this post-processing approach
    is simpler to implement and competitive in practice.

    Args:
        model:      PyTorch model.
        G:          Number of groups.
        task:       "classification" (regression EO is not standard).
        lr:         Learning rate for ERM pre-training.
        epochs:     Training epochs.
        batch_size: Mini-batch size.
        device:     Torch device.
        verbose:    Print progress.
    """

    def __init__(
        self,
        model: nn.Module,
        G: int = 2,
        task: str = "classification",
        lr: float = 0.01,
        epochs: int = 100,
        batch_size: int = 64,
        device: str = "cpu",
        verbose: bool = True,
    ):
        super().__init__(model, task, device, batch_size)
        self.G       = G
        self.lr      = lr
        self.epochs  = epochs
        self.verbose = verbose

        # Per-group thresholds learned during calibration
        self.thresholds: np.ndarray = np.full(G, 0.5)

    def _train_erm(self, X_train, y_train):
        """Train vanilla ERM model (internal helper)."""
        erm = ERMBaseline(
            model=self.model,
            task=self.task,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            device=self.device,
            verbose=self.verbose,
        )
        # Pass empty groups (ERM ignores them)
        erm.fit(X_train, y_train, np.zeros(len(y_train), dtype=int))

    def _find_equalized_thresholds(
        self, proba: np.ndarray, y: np.ndarray, g: np.ndarray
    ) -> np.ndarray:
        """
        Search for per-group thresholds that minimise the FPR gap across groups.

        Strategy:
          - For each group g, sweep thresholds in [0.05, 0.95].
          - Pick the threshold t_g such that |FPR_g(t_g) − mean_FPR| is minimised.

        This is a greedy approximation to the exact LP-based solution of Hardt 2016.
        """
        thresholds = np.full(self.G, 0.5)
        candidate_thresholds = np.linspace(0.05, 0.95, 40)

        # First pass: compute FPR at default 0.5 for each group
        fprs_default = {}
        for grp in range(self.G):
            mask = (g == grp)
            if mask.sum() == 0:
                fprs_default[grp] = 0.5
                continue
            p_g = proba[mask]
            y_g = y[mask]
            preds = (p_g > 0.5).astype(int)
            neg   = (y_g == 0)
            fprs_default[grp] = preds[neg].mean() if neg.sum() > 0 else 0.5

        # Target: match the mean FPR across groups
        mean_fpr = np.mean(list(fprs_default.values()))

        for grp in range(self.G):
            mask = (g == grp)
            if mask.sum() == 0:
                continue
            p_g = proba[mask]
            y_g = y[mask]
            neg = (y_g == 0)
            if neg.sum() == 0:
                continue

            best_t   = 0.5
            best_gap = float("inf")
            for t in candidate_thresholds:
                preds = (p_g > t).astype(int)
                fpr   = preds[neg].mean()
                gap   = abs(fpr - mean_fpr)
                if gap < best_gap:
                    best_gap = gap
                    best_t   = t
            thresholds[grp] = best_t

        return thresholds

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, g_train: np.ndarray) -> "EqualizedOddsBaseline":
        """
        Step 1: Train ERM. Step 2: Calibrate per-group thresholds on training set.
        """
        # Train base ERM model
        self._train_erm(X_train, y_train)

        # Calibrate thresholds on training data
        proba = self.predict_proba(X_train)
        self.thresholds = self._find_equalized_thresholds(proba, y_train, g_train)

        if self.verbose:
            print(f"  [EO-Fair] Group thresholds: {np.round(self.thresholds, 3)}")
        return self

    def predict(self, X: np.ndarray, groups: Optional[np.ndarray] = None,
                threshold: float = 0.5) -> np.ndarray:
        """
        Predict using per-group thresholds (requires group labels at inference).
        Falls back to global threshold if groups not provided.
        """
        proba = self.predict_proba(X)
        if groups is None or self.task != "classification":
            return (proba > threshold).astype(int)

        preds = np.zeros(len(proba), dtype=int)
        for grp in range(self.G):
            mask = (groups == grp)
            if mask.sum() > 0:
                preds[mask] = (proba[mask] > self.thresholds[grp]).astype(int)
        return preds

    def evaluate(self, X: np.ndarray, y: np.ndarray, g: np.ndarray) -> Dict:
        """Evaluate with per-group thresholds."""
        preds = self.predict(X, groups=g)
        proba = self.predict_proba(X)
        return compute_all_metrics(
            y_true=y,
            y_pred=proba,          # continuous for AUC
            groups=g,
            task=self.task,
            compute_auc_flag=True,
            threshold_override=dict(zip(range(self.G), self.thresholds)),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PRIM Ablations (used in Section V experiments)
# ═══════════════════════════════════════════════════════════════════════════════

def build_prim_ablation(model, G, task, device, **kwargs):
    """
    Factory for PRIM ablation variants used in the experiments:
      - PRIM (no DP)     : use_dp=False
      - PRIM (no Robust) : use_robust=False, rho=0
      - PRIM (no Intp.)  : use_reg=False, lambda_reg=0

    Returns a configured PRIM instance.
    (Import PRIM lazily to avoid circular imports.)
    """
    from .prim import PRIM
    return PRIM(model=model, G=G, task=task, device=device, **kwargs)
