"""
baselines/erm.py — Empirical Risk Minimization (ERM) Baseline

Standard unconstrained training — minimizes overall error with no fairness constraints.
This is the simplest baseline and typically shows the largest fairness violations
(worst-group errors are highest) but best overall accuracy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fairness_metrics import compute_all_metrics


class ERM:
    """
    Standard Empirical Risk Minimization.

    Minimizes: (1/n) * Σ_i ℓ(f_θ(x_i), y_i)
    without any fairness, privacy, or robustness constraints.

    This baseline demonstrates the unfairness problem that PRIM solves.

    Args:
        model:     PyTorch model.
        task:      "classification" or "regression".
        lr:        Learning rate.
        n_epochs:  Number of training epochs.
        batch_size: Mini-batch size.
        device:    Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        task: str = "classification",
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 64,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.model = model.to(device)
        self.task = task
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        if task == "classification":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, g_train: np.ndarray = None) -> "ERM":
        """Train model using standard ERM (group labels g_train are ignored)."""
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(x_batch)
                loss = self.loss_fn(out, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"  ERM Epoch {epoch+1}/{self.n_epochs} | Loss={total_loss/len(loader):.4f}")

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
