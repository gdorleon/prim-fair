"""
dp_utils.py — Differential Privacy Utilities for PRIM
======================================================

Implements two DP mechanisms used in Algorithm 1:

  1. Noisy Group-Loss Queries (Step 4 of Algorithm 1)
     – Gaussian or Laplace mechanism on per-group loss values
     – Achieves (ε_g, δ_g)-DP for the group-selection step

  2. DP-SGD (Step 6 of Algorithm 1)
     – Per-sample gradient clipping (sensitivity = C / batch_size)
     – Gaussian noise calibrated to (ε_m, δ_m)-DP per step
     – Total privacy budget tracked via advanced composition

Privacy accounting references:
  - Dwork et al. (2014) "The Algorithmic Foundations of Differential Privacy"
    Theorem 3.20 (advanced composition)
  - Abadi et al. (2016) "Deep Learning with Differential Privacy"
    (moments accountant — tighter bound used by Opacus)
"""

import math
import numpy as np
from typing import Optional, Dict, List

# PyTorch is only needed for the DP-SGD functions (clip/add-noise/step).
# We import it lazily so the DP accounting and noise-calibration utilities
# can be used without a full PyTorch installation (e.g., in unit tests).
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:                            # pragma: no cover
    _TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  Basic DP noise calibration
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_noise_scale(sensitivity: float, epsilon: float, delta: float) -> float:
    """
    Compute Gaussian noise σ for (ε, δ)-DP using the analytic calibration.

    Formula (standard closed-form):
        σ = sensitivity * sqrt(2 * ln(1.25 / δ)) / ε

    This is a conservative upper bound; Opacus' moments accountant gives
    a tighter (smaller) σ in practice.

    Args:
        sensitivity: L2 sensitivity of the query (= clip_norm for DP-SGD).
        epsilon:     Privacy budget ε > 0.
        delta:       Failure probability 0 < δ < 1.

    Returns:
        sigma: Standard deviation of the Gaussian noise to add.
    """
    if epsilon <= 0 or not (0 < delta < 1):
        raise ValueError("Need ε > 0 and 0 < δ < 1")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def laplace_noise_scale(sensitivity: float, epsilon: float) -> float:
    """
    Compute Laplace noise scale b for (ε, 0)-DP.

    Formula: b = sensitivity / ε

    Args:
        sensitivity: L1 sensitivity of the query.
        epsilon:     Privacy budget ε > 0.

    Returns:
        b: Scale parameter of the Laplace distribution.
    """
    if epsilon <= 0:
        raise ValueError("Need ε > 0")
    return sensitivity / epsilon


def compute_dp_noise_sigma(
    target_epsilon: float,
    target_delta: float,
    num_iterations: int,
    sensitivity: float = 1.0,
    sample_rate: float = 1.0,
) -> float:
    """
    Compute the DP-SGD noise multiplier σ (as a multiple of sensitivity)
    such that, after `num_iterations` gradient steps, the total privacy cost
    satisfies (target_epsilon, target_delta)-DP via advanced composition.

    Uses binary search to find the largest per-step ε that composes safely.

    Args:
        target_epsilon:  Total desired ε.
        target_delta:    Total desired δ.
        num_iterations:  Total number of gradient-update steps T.
        sensitivity:     Gradient sensitivity = clip_norm (default 1.0).
        sample_rate:     Subsampling rate q = batch_size / dataset_size.
                         Subsampling amplifies privacy (makes ε smaller).

    Returns:
        sigma: Noise scale σ such that adding N(0, σ²) noise per step
               achieves (target_epsilon, target_delta)-DP overall.
    """
    T    = num_iterations
    q    = sample_rate
    delt = target_delta

    def composed_epsilon(eps_step: float) -> float:
        """
        Advanced composition (Dwork 2014, Theorem 3.20) with
        subsampling amplification: ε_amp = q * ε_step.

        ε_total = sqrt(2T * ln(1/δ)) * ε_amp + T * ε_amp * (e^ε_amp − 1)
        """
        eps_amp = q * eps_step
        term1   = math.sqrt(2 * T * math.log(1.0 / delt)) * eps_amp
        term2   = T * eps_amp * (math.exp(eps_amp) - 1.0)
        return term1 + term2

    # Binary-search for the largest per-step ε that stays within budget
    lo, hi = 1e-8, 20.0
    for _ in range(200):        # 200 bisection steps → ~60 bits of precision
        mid = 0.5 * (lo + hi)
        if composed_epsilon(mid) <= target_epsilon:
            lo = mid
        else:
            hi = mid
    eps_step = lo

    # Convert per-step ε to Gaussian σ (each step is (eps_step, δ/T)-DP)
    delta_step = delt / T
    sigma = gaussian_noise_scale(sensitivity, eps_step, delta_step)
    return sigma


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  Noisy group-loss evaluation (Algorithm 1, Step 4)
# ═══════════════════════════════════════════════════════════════════════════════

def privatize_group_losses(
    group_losses: np.ndarray,
    sensitivity: float,
    epsilon_g: float,
    mechanism: str = "gaussian",
    delta_g: float = 1e-6,
) -> np.ndarray:
    """
    Add calibrated DP noise to per-group loss estimates so that
    the group-query step of PRIM satisfies (ε_g, δ_g)-DP.

    From Algorithm 1:
        L̃_g = L_g(θ^(t-1)) + Laplace(b)  or  N(0, b²)

    Args:
        group_losses: Clean per-group losses, shape (G,), values in [0, 1].
        sensitivity:  L1/L2 sensitivity of the group-loss query.
                      For a mean of bounded losses ℓ ∈ [0, 1]:
                        L1 sensitivity ≈ 1 / n_g
                        Conservative upper bound: 1.0
        epsilon_g:    Privacy budget for the group query step.
        mechanism:    "laplace" (exact DP) or "gaussian" (approximate DP).
        delta_g:      Failure probability (only used for Gaussian).

    Returns:
        noisy_losses: Privatized per-group losses, shape (G,).
                      Clipped to [0, 1] since losses are bounded.
    """
    G = len(group_losses)
    noisy = np.copy(group_losses)

    if mechanism == "laplace":
        # (ε_g, 0)-DP: exact DP guarantee
        b = laplace_noise_scale(sensitivity, epsilon_g)
        noise = np.random.laplace(0.0, b, size=G)
    elif mechanism == "gaussian":
        # (ε_g, δ_g)-DP: approximate DP (tighter noise for same ε)
        sigma = gaussian_noise_scale(sensitivity, epsilon_g, delta_g)
        noise = np.random.normal(0.0, sigma, size=G)
    else:
        raise ValueError(f"Unknown DP mechanism: '{mechanism}'. Use 'laplace' or 'gaussian'.")

    noisy += noise
    # Clip to valid loss range — negative losses don't make sense
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  DP-SGD (Algorithm 1, Step 6)
# ═══════════════════════════════════════════════════════════════════════════════

def clip_per_sample_gradients(model, max_grad_norm: float) -> None:
    """
    Per-sample gradient clipping (Step 1 of DP-SGD).

    Standard torch.autograd gives *batch-averaged* gradients.
    For DP-SGD we need to clip *each sample's* gradient individually,
    then aggregate. We approximate this by clipping the total gradient
    by max_grad_norm (valid when batch_size=1, or as an approximation).

    For exact per-sample clipping use Opacus (privacy engine wraps DataLoader).

    This clips: g ← g * min(1, C / ‖g‖₂)
    """
    total_norm = 0.0
    params_with_grads = [p for p in model.parameters() if p.grad is not None]

    # Compute total gradient norm across all parameters
    for p in params_with_grads:
        total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = math.sqrt(total_norm)

    # Scale factor: 1 if norm ≤ C, else C/norm
    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params_with_grads:
            p.grad.data.mul_(clip_coef)


def add_gaussian_noise_to_gradients(
    model,
    sigma: float,
    max_grad_norm: float,
    batch_size: int,
) -> None:
    """
    Add Gaussian noise to clipped gradients (Step 2 of DP-SGD).

    After clipping, we add N(0, σ²C²) noise to the *sum* of gradients,
    then divide by batch_size to get the noisy *mean* gradient.

    Actual noise scale on the averaged gradient:
        σ_effective = σ * C / batch_size

    where C = max_grad_norm, σ = noise_multiplier.

    Args:
        model:         Model with .grad populated after backward().
        sigma:         Noise multiplier (σ in DP-SGD; larger = more privacy).
        max_grad_norm: Gradient clipping threshold C.
        batch_size:    Mini-batch size (for normalisation).
    """
    noise_scale = sigma * max_grad_norm / batch_size
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                # Sample noise from N(0, noise_scale²) with the same shape as gradient
                noise = torch.randn_like(p.grad) * noise_scale
                p.grad.add_(noise)


def dp_sgd_step(
    model,
    optimizer,  # torch.optim.Optimizer
    loss,       # torch.Tensor
    sigma: float,
    max_grad_norm: float,
    batch_size: int,
) -> None:
    """
    One complete DP-SGD gradient update:
        1. Zero gradients.
        2. Compute gradients via backward().
        3. Clip gradients to L2 norm ≤ max_grad_norm.
        4. Add calibrated Gaussian noise.
        5. Step the optimizer.

    This is the private variant of the standard optimizer.step() call
    used in Step 6 of Algorithm 1.

    Args:
        model:         PyTorch model being trained.
        optimizer:     SGD / Adam optimizer.
        loss:          Scalar loss tensor (from forward pass).
        sigma:         DP-SGD noise multiplier.
        max_grad_norm: Gradient clipping bound C.
        batch_size:    Current mini-batch size.
    """
    # 1. Clear old gradients
    optimizer.zero_grad()

    # 2. Compute ∂ℓ/∂θ via automatic differentiation
    loss.backward()

    # 3. Clip: g ← g * min(1, C / ‖g‖)
    clip_per_sample_gradients(model, max_grad_norm)

    # 4. Add noise: g ← g + N(0, (σC/n)²I)
    if sigma > 0:
        add_gaussian_noise_to_gradients(model, sigma, max_grad_norm, batch_size)

    # 5. Gradient step: θ ← θ - lr * g
    optimizer.step()


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  Privacy accounting (tracks ε spent over training)
# ═══════════════════════════════════════════════════════════════════════════════

class DPAccountant:
    """
    Tracks cumulative privacy cost across all training iterations.

    Uses the advanced composition theorem (Dwork 2014) to bound
    total ε after T steps, each consuming (ε_m, δ_m)-DP.

    For tighter accounting, substitute Opacus' RDP accountant, but
    this class provides a self-contained implementation for transparency.

    Args:
        target_epsilon:  Desired total ε budget.
        target_delta:    Desired total δ budget.
        num_iterations:  Total planned number of DP-SGD steps T.
    """

    def __init__(
        self,
        target_epsilon: float,
        target_delta: float,
        num_iterations: int,
    ):
        self.target_epsilon  = target_epsilon
        self.target_delta    = target_delta
        self.num_iterations  = num_iterations

        # Running total of privacy cost spent so far
        self._eps_spent: float = 0.0
        self._steps_done: int  = 0

        # Epsilon budget allocated to group queries (5% of total)
        self.epsilon_group = target_epsilon * 0.05

        # Epsilon budget for DP-SGD (95% of total)
        self.epsilon_model = target_epsilon * 0.95

    def per_step_epsilon(self) -> float:
        """
        Compute the per-step ε for DP-SGD such that after num_iterations
        steps, the total composition stays within (epsilon_model, delta)-DP.

        This inverts the composition formula via the same binary search
        used in compute_dp_noise_sigma.
        """
        T    = self.num_iterations
        delt = self.target_delta
        eps_total_budget = self.epsilon_model

        def composed(eps_step):
            term1 = math.sqrt(2 * T * math.log(1.0 / delt)) * eps_step
            term2 = T * eps_step * (math.exp(eps_step) - 1.0)
            return term1 + term2

        lo, hi = 1e-8, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if composed(mid) <= eps_total_budget:
                lo = mid
            else:
                hi = mid
        return lo

    def total_privacy_spent(self, eps_per_step: float) -> float:
        """
        Compute total ε after self._steps_done steps using advanced composition.

        Formula (Dwork 2014 Theorem 3.20):
          ε_total = sqrt(2T * ln(1/δ)) * ε_step + T * ε_step * (e^ε_step − 1)
                    + ε_group_queries
        """
        T    = self._steps_done
        delt = self.target_delta
        if T == 0 or eps_per_step == 0:
            return self.epsilon_group

        term1 = math.sqrt(2 * T * math.log(1.0 / max(delt, 1e-20))) * eps_per_step
        term2 = T * eps_per_step * (math.exp(eps_per_step) - 1.0)
        return term1 + term2 + self.epsilon_group

    def step(self) -> None:
        """Record that one DP-SGD step has been taken."""
        self._steps_done += 1

    def budget_remaining(self, eps_per_step: float) -> float:
        """How much ε remains in the budget."""
        return self.target_epsilon - self.total_privacy_spent(eps_per_step)

    def is_budget_exhausted(self, eps_per_step: float) -> bool:
        """True if continuing would exceed the privacy budget."""
        return self.total_privacy_spent(eps_per_step) >= self.target_epsilon

    def report(self, eps_per_step: float) -> Dict:
        """Return a dictionary summarizing the privacy accounting."""
        return {
            "target_epsilon"       : self.target_epsilon,
            "target_delta"         : self.target_delta,
            "steps_done"           : self._steps_done,
            "total_steps_planned"  : self.num_iterations,
            "epsilon_per_step"     : eps_per_step,
            "epsilon_group_queries": self.epsilon_group,
            "epsilon_spent_total"  : self.total_privacy_spent(eps_per_step),
            "delta_total"          : self.target_delta,
            "budget_remaining"     : self.budget_remaining(eps_per_step),
        }

    def print_report(self, eps_per_step: float) -> None:
        """Pretty-print the privacy accounting report."""
        r = self.report(eps_per_step)
        print("\n── DP Accounting ───────────────────────────────────────")
        print(f"  Steps done / planned: {r['steps_done']} / {r['total_steps_planned']}")
        print(f"  ε per DP-SGD step   : {r['epsilon_per_step']:.5f}")
        print(f"  ε for group queries : {r['epsilon_group_queries']:.4f}")
        print(f"  ε spent (total)     : {r['epsilon_spent_total']:.4f}")
        print(f"  ε remaining         : {r['budget_remaining']:.4f}")
        print(f"  δ total             : {r['delta_total']:.2e}")
        is_ok = r['epsilon_spent_total'] <= r['target_epsilon']
        print(f"  Budget status       : {'✓ WITHIN BUDGET' if is_ok else '✗ OVER BUDGET'}")
        print("────────────────────────────────────────────────────────\n")
