"""
robustness.py — Adversarial Robustness via PGD for PRIM

This module implements:
  1. PGD (Projected Gradient Descent) attack for ℓ∞-bounded perturbations
  2. Robust loss computation: E[max_{‖Δ‖∞ ≤ ρ} ℓ(f(x+Δ), y)]
  3. Adversarial batch generation for training

Reference:
  Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018.
  Xu et al., adversarial robustness + fairness connection (cited in paper).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    rho: float,
    num_steps: int = 20,
    step_size: Optional[float] = None,
    random_start: bool = True,
) -> torch.Tensor:
    """
    PGD (Projected Gradient Descent) adversarial attack under ℓ∞ constraint.

    Finds the worst-case perturbation Δ such that ‖Δ‖∞ ≤ ρ that
    maximizes the loss ℓ(f(x + Δ), y).

    Algorithm (Madry et al. 2018):
        x_adv_0 = x + uniform_noise (if random_start)
        For step k:
            x_adv_{k+1} = Proj_{B∞(x, ρ)}[ x_adv_k + step_size * sign(∇_x loss) ]

    Args:
        model:       The model f_θ being attacked.
        x:           Input batch, shape (B, d).
        y:           True labels, shape (B,).
        loss_fn:     Loss function ℓ(predictions, labels) → scalar.
        rho:         ℓ∞ perturbation radius (paper's ρ parameter).
        num_steps:   Number of PGD iterations (paper uses 20).
        step_size:   Per-step attack step size. Defaults to rho * 2.5 / num_steps.
        random_start: Whether to start with random noise (improves attack quality).

    Returns:
        x_adv: Adversarial inputs of same shape as x.
    """
    if rho == 0.0:
        # No robustness required — return clean inputs unchanged
        return x.clone()

    # Default step size: proportional to rho, as commonly used in literature
    if step_size is None:
        step_size = rho * 2.5 / num_steps

    model.eval()  # Don't update BatchNorm stats during attack

    # Clone x so we don't modify the original; detach to start fresh gradient tracking
    x_adv = x.clone().detach()

    if random_start:
        # Random initialization within the ℓ∞ ball — helps escape local maxima
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-rho, rho)
        # Project back into the ℓ∞ ball around the original x
        x_adv = torch.clamp(x_adv, x - rho, x + rho)

    for step in range(num_steps):
        # Enable gradient computation on x_adv for this step
        x_adv.requires_grad_(True)

        # Forward pass with current adversarial example
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)

        # Compute gradient of loss w.r.t. adversarial input
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

        # FGSM step: move in the direction that maximizes loss (gradient ascent)
        # sign() gives the direction, step_size controls magnitude
        x_adv_new = x_adv.detach() + step_size * grad.sign()

        # Project back into ℓ∞ ball: clip element-wise to [x - rho, x + rho]
        x_adv = torch.clamp(x_adv_new, x - rho, x + rho).detach()

    model.train()  # Restore training mode
    return x_adv


def compute_robust_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    rho: float,
    num_pgd_steps: int = 20,
) -> torch.Tensor:
    """
    Compute the robust loss: E[max_{‖Δ‖∞ ≤ ρ} ℓ(f(x+Δ), y)]

    This is the inner maximization step of adversarial training.
    The outer minimization is handled by PRIM's optimizer.

    Args:
        model:         The model f_θ.
        x:             Input batch.
        y:             Labels.
        loss_fn:       Loss function.
        rho:           ℓ∞ perturbation radius.
        num_pgd_steps: PGD iterations (more = stronger attack but slower).

    Returns:
        robust_loss: Scalar tensor — the worst-case loss over perturbations.
    """
    # Find worst-case adversarial examples using PGD
    x_adv = pgd_attack(
        model=model,
        x=x,
        y=y,
        loss_fn=loss_fn,
        rho=rho,
        num_steps=num_pgd_steps,
    )

    # Evaluate model on adversarial examples (this is the loss we minimize)
    model.train()
    outputs = model(x_adv)
    robust_loss = loss_fn(outputs, y)

    return robust_loss


def compute_group_robust_losses(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    groups: torch.Tensor,
    loss_fn: Callable,
    rho: float,
    G: int,
    num_pgd_steps: int = 20,
) -> torch.Tensor:
    """
    Compute per-group robust losses for the PRIM group weight update.

    For each group g, computes:
        L_g^(rob)(θ) = (1/|D_g|) * sum_{(x,y) in D_g} max_{‖Δ‖∞ ≤ ρ} ℓ(f(x+Δ), y)

    Args:
        model:         The model f_θ.
        x:             Full batch inputs, shape (B, d).
        y:             Full batch labels, shape (B,).
        groups:        Group membership array, shape (B,), values in {0,...,G-1}.
        loss_fn:       Per-sample loss function (reduction='none').
        rho:           ℓ∞ perturbation radius.
        G:             Number of demographic groups.
        num_pgd_steps: Number of PGD steps.

    Returns:
        group_losses: Tensor of shape (G,) — robust loss per group.
    """
    # Generate adversarial examples for the whole batch at once (efficient)
    x_adv = pgd_attack(
        model=model,
        x=x,
        y=y,
        loss_fn=lambda preds, labels: loss_fn(preds, labels).mean(),
        rho=rho,
        num_steps=num_pgd_steps,
    )

    model.train()
    outputs = model(x_adv)

    # Compute per-sample losses (no reduction yet)
    sample_losses = loss_fn(outputs, y)  # shape: (B,)

    # Aggregate losses by group
    group_losses = torch.zeros(G, device=x.device)
    group_counts = torch.zeros(G, device=x.device)

    for g in range(G):
        mask = (groups == g)
        if mask.sum() > 0:
            group_losses[g] = sample_losses[mask].mean()
            group_counts[g] = mask.sum().float()
        else:
            # If a group has no samples in this batch, use 0 (won't affect weighting)
            group_losses[g] = torch.tensor(0.0, device=x.device)

    return group_losses, group_counts


def evaluate_robustness(
    model: nn.Module,
    data_loader,
    loss_fn: Callable,
    rho: float,
    groups_key: str = "group",
    G: int = 2,
    num_pgd_steps: int = 20,
    device: str = "cpu",
) -> dict:
    """
    Evaluate model robustness under PGD attack on a full dataset.

    Returns a dictionary with:
        - 'clean_loss': overall clean loss
        - 'robust_loss': overall adversarial loss
        - 'clean_group_losses': per-group clean losses
        - 'robust_group_losses': per-group adversarial losses
        - 'accuracy_clean': clean accuracy
        - 'accuracy_robust': adversarial accuracy
    """
    model.eval()
    clean_losses_all = []
    robust_losses_all = []
    groups_all = []
    clean_correct = 0
    robust_correct = 0
    total = 0

    for batch in data_loader:
        x_batch = batch["x"].to(device)
        y_batch = batch["y"].to(device)
        g_batch = batch[groups_key].to(device)

        # Clean evaluation
        with torch.no_grad():
            clean_out = model(x_batch)
            clean_loss = loss_fn(clean_out, y_batch)  # per-sample
            clean_losses_all.append(clean_loss.cpu())

            if clean_out.dim() > 1 and clean_out.shape[1] > 1:
                clean_pred = clean_out.argmax(dim=1)
                clean_correct += (clean_pred == y_batch).sum().item()
            else:
                clean_pred = (clean_out.squeeze() > 0).long()
                clean_correct += (clean_pred == y_batch).sum().item()

        # Adversarial evaluation
        x_adv = pgd_attack(
            model=model,
            x=x_batch,
            y=y_batch,
            loss_fn=lambda p, l: loss_fn(p, l).mean(),
            rho=rho,
            num_steps=num_pgd_steps,
        )
        with torch.no_grad():
            robust_out = model(x_adv)
            robust_loss = loss_fn(robust_out, y_batch)
            robust_losses_all.append(robust_loss.cpu())

            if robust_out.dim() > 1 and robust_out.shape[1] > 1:
                robust_pred = robust_out.argmax(dim=1)
                robust_correct += (robust_pred == y_batch).sum().item()
            else:
                robust_pred = (robust_out.squeeze() > 0).long()
                robust_correct += (robust_pred == y_batch).sum().item()

        groups_all.append(g_batch.cpu())
        total += x_batch.shape[0]

    # Concatenate everything
    all_clean = torch.cat(clean_losses_all)
    all_robust = torch.cat(robust_losses_all)
    all_groups = torch.cat(groups_all)

    results = {
        "clean_loss": all_clean.mean().item(),
        "robust_loss": all_robust.mean().item(),
        "accuracy_clean": clean_correct / total,
        "accuracy_robust": robust_correct / total,
        "clean_group_losses": {},
        "robust_group_losses": {},
    }

    for g in range(G):
        mask = (all_groups == g)
        if mask.sum() > 0:
            results["clean_group_losses"][g] = all_clean[mask].mean().item()
            results["robust_group_losses"][g] = all_robust[mask].mean().item()

    model.train()
    return results
