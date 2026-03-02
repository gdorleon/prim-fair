"""
experiments/run_marketing.py — Bank Marketing Experiment

Reproduces Table II: Impact of Differential Privacy on Marketing Dataset.

Reports AUC and Demographic Parity Gap for:
  - ERM (No DP)
  - ERM (ε=1 DP)
  - PRIM (No DP)
  - PRIM (ε=1 DP)

Usage:
    python experiments/run_marketing.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets import load_marketing, generate_synthetic_dataset
from src.models import build_model
from src.prim import PRIM
from src.fairness_metrics import compute_all_metrics, demographic_parity_gap, compute_auc
from baselines.erm import ERM

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_dp_impact_experiment(dataset: dict, device: str = "cpu") -> pd.DataFrame:
    """
    Reproduce Table II: Impact of Differential Privacy on Marketing Dataset.

    Four conditions:
      1. ERM (No DP): Standard training, no privacy
      2. ERM (ε=1 DP): Standard training + DP-SGD
      3. PRIM (No DP): Minimax fairness, no privacy
      4. PRIM (ε=1 DP): Full PRIM with (ε=1, δ=1e-5)-DP
    """
    print("\n" + "="*60)
    print("  Table II: Impact of Differential Privacy on Marketing")
    print("="*60)

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    g_train = dataset["g_train"]
    X_test  = dataset["X_test"]
    y_test  = dataset["y_test"]
    g_test  = dataset["g_test"]
    G       = dataset["n_groups"]
    d       = dataset["input_dim"]

    results_list = []

    # ---- 1. ERM (No DP) ----
    print("\n[1] ERM (No DP)...")
    model1 = build_model("linear", d, task="classification")
    erm_nodp = ERM(model1, lr=0.005, n_epochs=100, batch_size=64, device=device, verbose=False)
    erm_nodp.fit(X_train, y_train, g_train)
    proba1 = erm_nodp.predict_proba(X_test)
    auc1 = compute_auc(y_test, proba1) * 100
    dp_gap1 = demographic_parity_gap(proba1, g_test)
    print(f"  AUC={auc1:.1f}%, DP Gap={dp_gap1:.2f}")
    results_list.append({"Method": "ERM (No DP)", "AUC (%)": round(auc1, 1), "Demographic Parity Gap": round(dp_gap1, 2)})

    # ---- 2. ERM (ε=1 DP) — simulate DP by adding gradient noise ----
    # We implement this by using PRIM with DP enabled but no fairness constraints
    # (uniform group weights that never change = ERM + DP)
    print("\n[2] ERM (ε=1 DP)...")
    model2 = build_model("linear", d, task="classification")
    # Use PRIM with alpha=1.0 (no group reweighting) + DP = DP-ERM
    erm_dp = PRIM(
        model=model2, G=G, epsilon=1.0, delta=1e-5, rho=0.0,
        lambda_reg=0.0, T=100, lr=0.005, batch_size=64,
        max_grad_norm=1.0, alpha=1.0,  # alpha=1 → no group reweighting
        n_pgd_steps=0, use_dp=True, use_robust=False, use_reg=False,
        task="classification", device=device, verbose=False, log_every=100,
    )
    erm_dp.fit(X_train, y_train, g_train)
    proba2 = erm_dp.predict_proba(X_test)
    auc2 = compute_auc(y_test, proba2) * 100
    dp_gap2 = demographic_parity_gap(proba2, g_test)
    print(f"  AUC={auc2:.1f}%, DP Gap={dp_gap2:.2f}")
    results_list.append({"Method": "ERM (ε=1 DP)", "AUC (%)": round(auc2, 1), "Demographic Parity Gap": round(dp_gap2, 2)})

    # ---- 3. PRIM (No DP) ----
    print("\n[3] PRIM (No DP)...")
    model3 = build_model("linear", d, task="classification")
    prim_nodp = PRIM(
        model=model3, G=G, epsilon=1.0, delta=1e-5, rho=0.1,
        lambda_reg=0.01, T=100, lr=0.005, batch_size=64,
        max_grad_norm=1.0, alpha=1.2, n_pgd_steps=5,
        use_dp=False,  # ← No DP
        use_robust=True, use_reg=True,
        task="classification", device=device, verbose=False, log_every=100,
    )
    prim_nodp.fit(X_train, y_train, g_train)
    proba3 = prim_nodp.predict_proba(X_test)
    auc3 = compute_auc(y_test, proba3) * 100
    dp_gap3 = demographic_parity_gap(proba3, g_test)
    print(f"  AUC={auc3:.1f}%, DP Gap={dp_gap3:.2f}")
    results_list.append({"Method": "PRIM (No DP)", "AUC (%)": round(auc3, 1), "Demographic Parity Gap": round(dp_gap3, 2)})

    # ---- 4. PRIM (ε=1 DP) — Full PRIM ----
    print("\n[4] PRIM (ε=1 DP) — Full PRIM...")
    model4 = build_model("linear", d, task="classification")
    prim_dp = PRIM(
        model=model4, G=G, epsilon=1.0, delta=1e-5, rho=0.1,
        lambda_reg=0.01, T=100, lr=0.005, batch_size=64,
        max_grad_norm=1.0, alpha=1.2, n_pgd_steps=5,
        use_dp=True, use_robust=True, use_reg=True,
        task="classification", device=device, verbose=True, log_every=25,
    )
    prim_dp.fit(X_train, y_train, g_train)
    proba4 = prim_dp.predict_proba(X_test)
    auc4 = compute_auc(y_test, proba4) * 100
    dp_gap4 = demographic_parity_gap(proba4, g_test)
    print(f"  AUC={auc4:.1f}%, DP Gap={dp_gap4:.2f}")
    results_list.append({"Method": "PRIM (ε=1 DP)", "AUC (%)": round(auc4, 1), "Demographic Parity Gap": round(dp_gap4, 2)})

    df = pd.DataFrame(results_list)
    print("\nTable II: Impact of Differential Privacy on Marketing Dataset")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "table2_dp_impact_marketing.csv"), index=False)
    print(f"\nSaved to results/table2_dp_impact_marketing.csv")

    return df, prim_dp


def run_epsilon_sweep(dataset: dict, device: str = "cpu") -> None:
    """
    Additional experiment: sweep ε from 0.1 to 5.0 and report AUC and DP gap.
    Shows the privacy-fairness trade-off curve (referenced in Section V-B of paper).
    """
    print("\n" + "="*60)
    print("  ε Sweep: Privacy-Fairness Trade-off on Marketing")
    print("="*60)

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    g_train = dataset["g_train"]
    X_test  = dataset["X_test"]
    y_test  = dataset["y_test"]
    g_test  = dataset["g_test"]
    G       = dataset["n_groups"]
    d       = dataset["input_dim"]

    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    aucs = []
    dp_gaps = []
    wges = []

    for eps in epsilon_values:
        print(f"\n  Training PRIM with ε={eps}...", flush=True)
        model = build_model("linear", d, task="classification")
        prim = PRIM(
            model=model, G=G, epsilon=eps, delta=1e-5, rho=0.1,
            lambda_reg=0.01, T=60, lr=0.005, batch_size=64,
            max_grad_norm=1.0, alpha=1.2, n_pgd_steps=5,
            use_dp=True, use_robust=True, use_reg=True,
            task="classification", device=device, verbose=False, log_every=100,
        )
        prim.fit(X_train, y_train, g_train)
        proba = prim.predict_proba(X_test)
        metrics = compute_all_metrics(y_test, proba, g_test, "classification")
        aucs.append(metrics.get("auc", 0.5) * 100)
        dp_gaps.append(metrics["demographic_parity_gap"])
        wges.append(metrics["worst_group_error"] * 100)
        print(f"    AUC={aucs[-1]:.1f}%, DP Gap={dp_gaps[-1]:.3f}, WGE={wges[-1]:.1f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epsilon_values, aucs, "b-o", linewidth=2, markersize=8)
    axes[0].set_xlabel("Privacy Budget ε", fontsize=12)
    axes[0].set_ylabel("AUC (%)", fontsize=12)
    axes[0].set_title("AUC vs. Privacy Budget ε\n(Marketing Dataset)", fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale("log")

    axes[1].plot(epsilon_values, dp_gaps, "r-o", linewidth=2, markersize=8)
    axes[1].set_xlabel("Privacy Budget ε", fontsize=12)
    axes[1].set_ylabel("Demographic Parity Gap", fontsize=12)
    axes[1].set_title("Fairness vs. Privacy Budget ε\n(Marketing Dataset)", fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale("log")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "marketing_epsilon_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ ε sweep figure saved to {path}")

    # Save results
    df = pd.DataFrame({
        "epsilon": epsilon_values,
        "AUC (%)": [round(a, 1) for a in aucs],
        "DP Gap": [round(g, 3) for g in dp_gaps],
        "WGE (%)": [round(w, 1) for w in wges],
    })
    df.to_csv(os.path.join(RESULTS_DIR, "marketing_epsilon_sweep.csv"), index=False)


def main():
    print("\n" + "="*60)
    print("  PRIM-Fair: Bank Marketing Experiment")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    print("\nLoading Bank Marketing dataset...")
    try:
        dataset = load_marketing()
    except FileNotFoundError:
        print("  Marketing data not found. Using synthetic substitute.")
        print("  Run 'python data/download_datasets.py' for real data.")
        dataset = generate_synthetic_dataset(
            n_samples=5000, n_features=30, G=2, task="classification"
        )
        dataset["name"] = "Marketing (Synthetic)"

    print(f"  Dataset: {dataset['name']}")
    print(f"  Train: {dataset['n_train']}, Test: {dataset['n_test']}")
    print(f"  Features: {dataset['input_dim']}, Groups: {dataset['n_groups']}")

    # Run Table II
    df_table2, prim = run_dp_impact_experiment(dataset, device)

    # Run epsilon sweep
    run_epsilon_sweep(dataset, device)

    print("\n" + "="*60)
    print("  Marketing experiments complete!")
    print("="*60)


if __name__ == "__main__":
    main()
