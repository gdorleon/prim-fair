"""
baselines/group_dro.py — Group DRO (Distributionally Robust Optimization) Baseline

Implements the minimax group fairness baseline without DP, robustness, or interpretability.
This is equivalent to:
  - Diana et al., "Minimax Group Fairness", AISTATS 2021
  - Sagawa et al., "Distributionally Robust Neural Networks for Group Shifts", ICLR 2020


"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fairness_metrics import compute_all_metrics


class GroupDRO:
    """
    Group Distributionally Robust Optimization.

    Solves: min_θ max_{g=1,...,G} L_g(θ)

    via the two-player game:
      - Learner updates θ by gradient descent on group-weighted loss
      - Regulator updates group weights by exponential gradient ascent on group losses

    This is the direct predecessor of PRIM — without DP, robustness, or interpretability.

    Args:
        model:      PyTorch model.
        G:          Number of groups.
        task:       "classification" or "regression".
        lr:         Learning rate for model parameters.
        eta:        Step size for group weight exponential gradient ascent.
        T:          Number of training iterations.
        batch_size: Mini-batch size.
        device:     Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        G: int = 2,
        task: str = "classification",
        lr: float = 0.01,
        eta: float = 0.1,
        T: int = 100,
        batch_size: int = 64,
        device: str = "cpu",
        verbose: bool = False,
        log_every: int = 20,
    ):
        self.model = model.to(device)
        self.G = G
        self.task = task
        self.lr = lr
        self.eta = eta
        self.T = T
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.log_every = log_every

        # Uniform initialization of group weights
        self.group_weights = np.ones(G) / G

        if task == "classification":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.loss_fn = nn.MSELoss(reduction="none")

        self.history = {"iteration": [], "worst_group_error": [], "group_weights": []}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        g_train: np.ndarray,
    ) -> "GroupDRO":
        """
        Train using Group DRO: alternate between group weight ascent and model descent.
        """
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        g_t = torch.tensor(g_train, dtype=torch.long)

        dataset = TensorDataset(X_t, y_t, g_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        if self.verbose:
            print(f"Training Group DRO: G={self.G}, T={self.T} iterations")

        for t in range(1, self.T + 1):
            self.model.train()

            for x_batch, y_batch, g_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                g_batch = g_batch.to(self.device)

                # Compute per-sample loss
                out = self.model(x_batch)
                per_sample_loss = self.loss_fn(out, y_batch)  # (B,)

                # Compute per-group loss
                group_losses_batch = torch.zeros(self.G, device=self.device)
                for g in range(self.G):
                    mask = (g_batch == g)
                    if mask.sum() > 0:
                        group_losses_batch[g] = per_sample_loss[mask].mean()

                # ---- Regulator: Exponential gradient ascent on group weights ----
                # w_g ← w_g * exp(η * L_g) — up-weight worse groups exponentially
                weights = torch.tensor(self.group_weights, dtype=torch.float32, device=self.device)
                weights = weights * torch.exp(self.eta * group_losses_batch.detach())
                weights = weights / weights.sum()  # project to simplex
                self.group_weights = weights.cpu().numpy()

                # ---- Learner: Gradient descent on group-weighted loss ----
                # Loss = Σ_g w_g * L_g(θ)
                sample_weights = torch.tensor(
                    [self.group_weights[g.item()] for g in g_batch],
                    dtype=torch.float32, device=self.device
                )
                weighted_loss = (sample_weights * per_sample_loss).sum() / (sample_weights.sum() + 1e-10)

                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()

            if t % self.log_every == 0 and self.verbose:
                # Evaluate
                self.model.eval()
                with torch.no_grad():
                    out = self.model(X_t.to(self.device))
                    if self.task == "classification":
                        preds = (torch.sigmoid(out) > 0.5).long().cpu().numpy()
                    else:
                        preds = out.cpu().numpy()

                group_errors = {}
                for g in range(self.G):
                    mask = (g_train == g)
                    if mask.sum() > 0:
                        if self.task == "classification":
                            group_errors[g] = 1 - (preds[mask] == y_train[mask]).mean()
                        else:
                            group_errors[g] = np.abs(preds[mask] - y_train[mask]).mean()

                wge = max(group_errors.values())
                print(f"  GroupDRO Iter {t}/{self.T} | WGE={wge:.4f} | Weights={np.round(self.group_weights, 3)}")
                self.history["iteration"].append(t)
                self.history["worst_group_error"].append(wge)
                self.history["group_weights"].append(self.group_weights.copy())
                self.model.train()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)
            if self.task == "classification":
                proba = torch.sigmoid(out).cpu().numpy()
            else:
                proba = out.cpu().numpy()
        self.model.train()
        return proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        if self.task == "classification":
            return (proba > threshold).astype(int)
        return proba

    def evaluate(self, X: np.ndarray, y: np.ndarray, g: np.ndarray) -> Dict:
        predictions = self.predict_proba(X)
        return compute_all_metrics(y, predictions, g, self.task,
                                   compute_auc_flag=(self.task == "classification"))
