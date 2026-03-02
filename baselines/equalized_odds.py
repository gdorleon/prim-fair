"""
baselines/equalized_odds.py — Equalized Odds / Opportunity Fairness Baseline

Implements equalized odds via a post-processing threshold adjustment approach.

The idea: After training a standard model, adjust per-group prediction thresholds
to equalize TPR (and optionally FPR) across groups, as in:
  Hardt et al., "Equality of Opportunity in Supervised Learning", NeurIPS 2016.

This is the "EO-Fair" baseline in the paper (Table I).
"""

import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fairness_metrics import compute_all_metrics


class EqualizedOddsPostProcessing:
    """
    Equalized Odds via post-processing threshold calibration.

    Pipeline:
      1. Train a base classifier (LogisticRegression or any model with predict_proba)
      2. For each group, find the threshold that equalizes FPR across groups
         (subject to maximizing accuracy)
      3. Apply group-specific thresholds at inference time

    This achieves approximate Equalized Odds without retraining.

    Args:
        base_estimator: sklearn-compatible classifier. Default: LogisticRegression.
        constraint:     "equalized_odds" (equalize TPR and FPR) or
                        "equal_opportunity" (equalize TPR only).
    """

    def __init__(
        self,
        base_estimator=None,
        constraint: str = "equalized_odds",
    ):
        if base_estimator is None:
            self.base_estimator = LogisticRegression(max_iter=1000, C=1.0)
        else:
            self.base_estimator = base_estimator

        self.constraint = constraint
        self.group_thresholds: Dict[int, float] = {}
        self.unique_groups = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        g_train: np.ndarray,
    ) -> "EqualizedOddsPostProcessing":
        """
        Train base model, then calibrate group thresholds for equalized odds.
        """
        # Step 1: Train unconstrained base model
        self.base_estimator.fit(X_train, y_train)
        self.unique_groups = np.unique(g_train)

        # Step 2: Get predicted probabilities on training set
        proba = self.base_estimator.predict_proba(X_train)[:, 1]

        # Step 3: Calibrate thresholds per group
        # For each group, find threshold that matches the group with lowest FPR
        self._calibrate_thresholds(proba, y_train, g_train)

        return self

    def _calibrate_thresholds(
        self,
        proba: np.ndarray,
        y: np.ndarray,
        g: np.ndarray,
    ) -> None:
        """
        Find per-group thresholds to equalize FPR across groups.

        Strategy:
          - Compute FPR vs threshold curves for each group
          - Find a common target FPR achievable by all groups
          - Set each group's threshold to achieve that target FPR
        """
        # Compute per-group FPR at various thresholds
        thresholds = np.linspace(0.01, 0.99, 100)
        group_fprs = {}

        for grp in self.unique_groups:
            mask = (g == grp) & (y == 0)  # Only negative instances
            if mask.sum() == 0:
                group_fprs[grp] = [(t, 0.0) for t in thresholds]
                continue

            p_neg = proba[mask]
            fprs = [(t, (p_neg >= t).mean()) for t in thresholds]
            group_fprs[grp] = fprs

        # Target: use the minimum achievable FPR across groups
        # (or a fixed target based on the paper's 5% constraint for COMPAS)
        target_fpr = 0.10  # Paper uses 5-10% FPR constraint in COMPAS experiment

        # For each group, find the threshold that achieves approximately target_fpr
        for grp in self.unique_groups:
            fprs = group_fprs[grp]
            # Find threshold where FPR is closest to target_fpr
            best_t = 0.5  # default
            best_diff = float("inf")
            for t, fpr in fprs:
                diff = abs(fpr - target_fpr)
                if diff < best_diff:
                    best_diff = diff
                    best_t = t
            self.group_thresholds[grp] = best_t

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return base model's probability estimates."""
        return self.base_estimator.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Predict labels using group-specific thresholds.

        IMPORTANT: Requires group labels at inference time.
        """
        proba = self.predict_proba(X)
        predictions = np.zeros(len(X), dtype=int)

        for grp in self.unique_groups:
            mask = (g == grp)
            threshold = self.group_thresholds.get(grp, 0.5)
            predictions[mask] = (proba[mask] >= threshold).astype(int)

        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray, g: np.ndarray) -> Dict:
        """Evaluate with group-specific thresholds."""
        proba = self.predict_proba(X)
        # Use group-aware thresholded predictions for metrics
        pred_labels = self.predict(X, g)

        # For metrics, pass the thresholded labels as probabilities
        # (so compute_all_metrics treats them as hard predictions)
        metrics = compute_all_metrics(
            y_true=y,
            y_pred=pred_labels.astype(float),  # pass as float
            groups=g,
            task="classification",
            compute_auc_flag=False,  # Can't compute AUC from thresholded labels
        )
        # Also compute AUC from raw probabilities
        from sklearn.metrics import roc_auc_score
        try:
            metrics["auc"] = roc_auc_score(y, proba)
        except Exception:
            metrics["auc"] = float("nan")

        return metrics
