# PRIM-Fair package
# Exposes the main algorithm and utilities for easy import
from .prim import PRIM
from .models import LinearModel, NeuralModelWithAttention, MixtureOfExpertsModel
from .dp_utils import DPAccountant, compute_dp_noise_sigma
from .robustness import pgd_attack, compute_robust_loss
from .fairness_metrics import (
    worst_group_error,
    overall_accuracy,
    demographic_parity_gap,
    equalized_odds_gap,
    compute_all_metrics,
)
from .datasets import (
    load_compas,
    load_communities_crime,
    load_bike_sharing,
    load_marketing,
    load_internet_traffic,
)

__all__ = [
    "PRIM",
    "LinearModel",
    "NeuralModelWithAttention",
    "MixtureOfExpertsModel",
    "DPAccountant",
    "compute_dp_noise_sigma",
    "pgd_attack",
    "compute_robust_loss",
    "worst_group_error",
    "overall_accuracy",
    "demographic_parity_gap",
    "equalized_odds_gap",
    "compute_all_metrics",
    "load_compas",
    "load_communities_crime",
    "load_bike_sharing",
    "load_marketing",
    "load_internet_traffic",
]
