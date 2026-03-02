"""
fairness_metrics.py — Fairness Evaluation Metrics for PRIM

Implements:
  - Worst-group error (main minimax metric)
  - Overall accuracy / MSE
  - Demographic parity gap
  - Equalized odds gap (true positive rate parity)
  - AUC and AUC gap
  - Per-group false positive rates (for COMPAS Table I)

Reference:
  Hardt et al., "Equality of Opportunity in Supervised Learning", NeurIPS 2016.
  Diana et al., "Minimax Group Fairness", AISTATS 2021.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Core Group Error Functions
# ---------------------------------------------------------------------------

def worst_group_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    task: str = "classification",
) -> Tuple[float, int]:
    """
    Compute the worst-case (maximum) error across all demographic groups.

    This is the primary fairness metric in the minimax framework:
        max_g E_{(x,y)~D_g}[ℓ(f(x), y)]

    Args:
        y_true:  True labels, shape (N,).
        y_pred:  Predicted labels (or probabilities for classification).
        groups:  Group membership, shape (N,), values in {0,...,G-1}.
        task:    "classification" (0/1 error) or "regression" (MAE).

    Returns:
        (worst_error, worst_group_idx): The maximum group error and which group it is.
    """
    unique_groups = np.unique(groups)
    group_errors = {}

    for g in unique_groups:
        mask = (groups == g)
        y_g = y_true[mask]
        p_g = y_pred[mask]

        if task == "classification":
            # Error rate = fraction of incorrect predictions
            if p_g.ndim > 1:
                pred_labels = p_g.argmax(axis=1)
            else:
                pred_labels = (p_g > 0.5).astype(int)
            group_errors[g] = 1.0 - (pred_labels == y_g).mean()

        elif task == "regression":
            # Mean Absolute Error for regression tasks
            group_errors[g] = mean_absolute_error(y_g, p_g)

    worst_group = max(group_errors, key=group_errors.get)
    return group_errors[worst_group], worst_group


def overall_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "classification",
) -> float:
    """
    Compute overall accuracy (classification) or 1 - MAE (regression proxy).

    Args:
        y_true: True labels.
        y_pred: Predictions (labels or probabilities).
        task:   "classification" or "regression".

    Returns:
        Accuracy (classification) or MAE (regression).
    """
    if task == "classification":
        if y_pred.ndim > 1:
            pred_labels = y_pred.argmax(axis=1)
        else:
            pred_labels = (y_pred > 0.5).astype(int)
        return (pred_labels == y_true).mean()

    elif task == "regression":
        return mean_absolute_error(y_true, y_pred)


def per_group_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    task: str = "classification",
) -> Dict[int, float]:
    """
    Compute error for each group separately.

    Returns:
        Dict mapping group_id → error rate / MAE.
    """
    unique_groups = np.unique(groups)
    errors = {}

    for g in unique_groups:
        mask = (groups == g)
        y_g = y_true[mask]
        p_g = y_pred[mask]

        if task == "classification":
            if p_g.ndim > 1:
                pred_labels = p_g.argmax(axis=1)
            else:
                pred_labels = (p_g > 0.5).astype(int)
            errors[g] = 1.0 - (pred_labels == y_g).mean()
        else:
            errors[g] = mean_absolute_error(y_g, p_g)

    return errors


# ---------------------------------------------------------------------------
# Fairness-Specific Metrics
# ---------------------------------------------------------------------------

def demographic_parity_gap(
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    Compute the Demographic Parity (Statistical Parity) gap.

    DP gap = max_g P(f(x)=1 | g) - min_g P(f(x)=1 | g)

    A gap of 0 means equal positive prediction rates across all groups.
    The paper reports DP gaps for the Marketing dataset (Table II).

    Args:
        y_pred:  Predicted probabilities or binary labels, shape (N,).
        groups:  Group membership, shape (N,).

    Returns:
        gap: The demographic parity gap (max - min positive rate).
    """
    unique_groups = np.unique(groups)
    positive_rates = []

    for g in unique_groups:
        mask = (groups == g)
        p_g = y_pred[mask]

        # Convert probabilities to binary predictions
        if p_g.ndim > 1:
            preds = p_g.argmax(axis=1)
        else:
            preds = (p_g > 0.5).astype(float)

        positive_rates.append(preds.mean())

    return max(positive_rates) - min(positive_rates)


def false_positive_rate_per_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[int, float]:
    """
    Compute False Positive Rate (FPR) for each group.

    FPR_g = P(f(x)=1 | y=0, group=g)

    Used in COMPAS experiments to measure racial bias in recidivism predictions.
    A high FPR for a minority group means false accusations of recidivism.

    Args:
        y_true:  True labels (0=no reoffend, 1=reoffend).
        y_pred:  Predicted probabilities or labels.
        groups:  Group membership.

    Returns:
        Dict mapping group_id → false positive rate.
    """
    unique_groups = np.unique(groups)
    fprs = {}

    for g in unique_groups:
        mask = (groups == g)
        y_g = y_true[mask]
        p_g = y_pred[mask]

        if p_g.ndim > 1:
            pred_labels = p_g.argmax(axis=1)
        else:
            pred_labels = (p_g > 0.5).astype(int)

        # FPR = FP / (FP + TN) = fraction of actual negatives predicted positive
        negatives_mask = (y_g == 0)
        if negatives_mask.sum() > 0:
            fprs[g] = pred_labels[negatives_mask].mean()
        else:
            fprs[g] = 0.0

    return fprs


def maximum_fpr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Compute max FPR across all groups (used for Table I in paper)."""
    fprs = false_positive_rate_per_group(y_true, y_pred, groups)
    return max(fprs.values())


def equalized_odds_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    Equalized Odds gap: measures difference in both TPR and FPR across groups.

    EO gap = max over groups of (|TPR_g - TPR_h| + |FPR_g - FPR_h|) for g ≠ h.

    Args:
        y_true:  True labels.
        y_pred:  Predicted probabilities.
        groups:  Group membership.

    Returns:
        gap: Maximum EO violation.
    """
    unique_groups = np.unique(groups)
    tprs = {}
    fprs = {}

    for g in unique_groups:
        mask = (groups == g)
        y_g = y_true[mask]
        p_g = y_pred[mask]

        if p_g.ndim > 1:
            pred = p_g.argmax(axis=1)
        else:
            pred = (p_g > 0.5).astype(int)

        pos_mask = (y_g == 1)
        neg_mask = (y_g == 0)

        tprs[g] = pred[pos_mask].mean() if pos_mask.sum() > 0 else 0.0
        fprs[g] = pred[neg_mask].mean() if neg_mask.sum() > 0 else 0.0

    # Compute pairwise differences
    max_gap = 0.0
    groups_list = list(unique_groups)
    for i in range(len(groups_list)):
        for j in range(i + 1, len(groups_list)):
            g1, g2 = groups_list[i], groups_list[j]
            gap = abs(tprs[g1] - tprs[g2]) + abs(fprs[g1] - fprs[g2])
            max_gap = max(max_gap, gap)

    return max_gap


def compute_auc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """
    Compute ROC-AUC. Returns NaN if only one class present.

    Args:
        y_true:       True binary labels.
        y_pred_proba: Predicted probabilities for positive class.

    Returns:
        auc: Area under ROC curve.
    """
    try:
        if y_pred_proba.ndim > 1:
            proba = y_pred_proba[:, 1]
        else:
            proba = y_pred_proba
        return roc_auc_score(y_true, proba)
    except ValueError:
        return float("nan")


def auc_gap_across_groups(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Compute max AUC - min AUC across groups (AUC parity gap)."""
    unique_groups = np.unique(groups)
    aucs = []

    for g in unique_groups:
        mask = (groups == g)
        auc_g = compute_auc(y_true[mask], y_pred_proba[mask])
        if not np.isnan(auc_g):
            aucs.append(auc_g)

    if len(aucs) < 2:
        return 0.0
    return max(aucs) - min(aucs)


# ---------------------------------------------------------------------------
# Comprehensive Metric Computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    task: str = "classification",
    compute_auc_flag: bool = True,
) -> Dict[str, float]:
    """
    Compute all fairness and accuracy metrics in one call.

    Returns a dictionary with:
        - 'worst_group_error':       Max error across groups
        - 'worst_group_idx':         Which group has worst error
        - 'overall_accuracy':        Mean accuracy (or MAE for regression)
        - 'per_group_errors':        Dict of {group: error}
        - 'demographic_parity_gap':  DP gap (classification only)
        - 'equalized_odds_gap':      EO gap (classification only)
        - 'auc':                     Overall AUC (classification only)
        - 'max_fpr':                 Max FPR (classification only)
        - 'fpr_per_group':           Dict of {group: FPR} (classification only)

    Args:
        y_true:   True labels.
        y_pred:   Predictions (probabilities for classification, values for regression).
        groups:   Group membership.
        task:     "classification" or "regression".
        compute_auc_flag: Whether to compute AUC (requires probability predictions).
    """
    metrics = {}

    # 1. Worst-group error (core minimax metric)
    wge, worst_g = worst_group_error(y_true, y_pred, groups, task)
    metrics["worst_group_error"] = wge
    metrics["worst_group_idx"] = worst_g

    # 2. Overall accuracy / MAE
    metrics["overall_accuracy"] = overall_accuracy(y_true, y_pred, task)

    # 3. Per-group breakdown
    metrics["per_group_errors"] = per_group_errors(y_true, y_pred, groups, task)

    if task == "classification":
        # 4. Fairness gaps
        metrics["demographic_parity_gap"] = demographic_parity_gap(y_pred, groups)
        metrics["equalized_odds_gap"] = equalized_odds_gap(y_true, y_pred, groups)

        # 5. AUC (if predictions are probabilities)
        if compute_auc_flag:
            metrics["auc"] = compute_auc(y_true, y_pred)

        # 6. False positive rates
        metrics["fpr_per_group"] = false_positive_rate_per_group(y_true, y_pred, groups)
        metrics["max_fpr"] = maximum_fpr(y_true, y_pred, groups)

    return metrics


def print_metrics(metrics: Dict, dataset_name: str = "", method_name: str = "PRIM") -> None:
    """Pretty-print all computed metrics for a given experiment."""
    print(f"\n{'='*60}")
    print(f"  Results: {method_name} on {dataset_name}")
    print(f"{'='*60}")
    print(f"  Worst-Group Error:       {metrics.get('worst_group_error', 'N/A'):.4f}")
    print(f"  Overall Accuracy:        {metrics.get('overall_accuracy', 'N/A'):.4f}")

    if "auc" in metrics:
        print(f"  AUC:                     {metrics.get('auc', 'N/A'):.4f}")

    if "demographic_parity_gap" in metrics:
        print(f"  Demographic Parity Gap:  {metrics.get('demographic_parity_gap', 'N/A'):.4f}")

    if "equalized_odds_gap" in metrics:
        print(f"  Equalized Odds Gap:      {metrics.get('equalized_odds_gap', 'N/A'):.4f}")

    if "max_fpr" in metrics:
        print(f"  Max FPR:                 {metrics.get('max_fpr', 'N/A'):.4f}")

    if "per_group_errors" in metrics:
        print(f"\n  Per-Group Errors:")
        for g, err in sorted(metrics["per_group_errors"].items()):
            print(f"    Group {g}: {err:.4f}")

    print(f"{'='*60}\n")
