"""
experiments/run_compas.py — COMPAS Recidivism Experiment

Reproduces:
  - Table I:   Fairness–Accuracy Trade-off on COMPAS
  - Table III: Robustness to Adversarial Attacks on COMPAS
  - Table IV:  Model Interpretability and Complexity on COMPAS
  - Figure 1:  Pareto frontier (Fairness-Error trade-off)

Usage:
    python experiments/run_compas.py
    python experiments/run_compas.py --plot
    python experiments/run_compas.py --robustness
    python experiments/run_compas.py --interpretability
    python experiments/run_compas.py --full   # Run all
"""

import sys
import os
import argparse
import numpy as np
import copy

# Make sure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import pandas as pd

from src.datasets import load_compas, generate_synthetic_dataset
from src.models import build_model
from src.prim import PRIM
from src.fairness_metrics import (
    compute_all_metrics, print_metrics, per_group_errors,
    false_positive_rate_per_group, maximum_fpr, compute_auc
)
from src.robustness import evaluate_robustness
from baselines.erm import ERM
from baselines.group_dro import GroupDRO
from baselines.equalized_odds import EqualizedOddsPostProcessing

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_table_I_fairness_accuracy(dataset: dict, device: str = "cpu") -> pd.DataFrame:
    """
    Reproduce Table I: Fairness–Accuracy Trade-off on COMPAS.

    Trains four methods and reports:
      - MGE: Maximum Group Error (%)
      - OA:  Overall Accuracy (%)
      - MFPR: Maximum FPR under 5% constraint
      - OE:  Overall Error under FPR constraint (%)
    """
    print("\n" + "="*60)
    print("  Table I: Fairness–Accuracy Trade-off on COMPAS")
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

    # ---- Method 1: ERM (unconstrained) ----
    print("\nTraining ERM...")
    erm_model = build_model("linear", d, task="classification")
    erm = ERM(erm_model, task="classification", lr=0.01, n_epochs=200, batch_size=64, device=device)
    erm.fit(X_train, y_train, g_train)
    erm_metrics = erm.evaluate(X_test, y_test, g_test)
    erm_mge = erm_metrics["worst_group_error"] * 100
    erm_oa  = erm_metrics["overall_accuracy"] * 100
    erm_mfpr = maximum_fpr(y_test, erm.predict_proba(X_test), g_test) * 100
    erm_oe  = (1 - erm_metrics["overall_accuracy"]) * 100
    results_list.append({
        "Method": "ERM", "MGE": erm_mge, "OA": erm_oa,
        "MFPR": erm_mfpr, "OE": erm_oe
    })
    print_metrics(erm_metrics, "COMPAS", "ERM")

    # ---- Method 2: Equalized Odds (EO-Fair) ----
    print("\nTraining Equalized Odds (EO-Fair)...")
    from sklearn.linear_model import LogisticRegression
    eo = EqualizedOddsPostProcessing(constraint="equalized_odds")
    eo.fit(X_train, y_train, g_train)
    eo_metrics = eo.evaluate(X_test, y_test, g_test)
    eo_mge  = eo_metrics["worst_group_error"] * 100
    eo_oa   = eo_metrics["overall_accuracy"] * 100
    eo_mfpr = maximum_fpr(y_test, eo.predict(X_test, g_test).astype(float), g_test) * 100
    eo_oe   = (1 - eo_metrics["overall_accuracy"]) * 100
    results_list.append({
        "Method": "EO‑Fair", "MGE": eo_mge, "OA": eo_oa,
        "MFPR": eo_mfpr, "OE": eo_oe
    })
    print_metrics(eo_metrics, "COMPAS", "EO-Fair")

    # ---- Method 3: Group DRO ----
    print("\nTraining Group DRO...")
    dro_model = build_model("linear", d, task="classification")
    dro = GroupDRO(dro_model, G=G, task="classification", lr=0.01, T=100,
                   batch_size=64, device=device, verbose=False, log_every=50)
    dro.fit(X_train, y_train, g_train)
    dro_metrics = dro.evaluate(X_test, y_test, g_test)
    dro_mge  = dro_metrics["worst_group_error"] * 100
    dro_oa   = dro_metrics["overall_accuracy"] * 100
    dro_mfpr = maximum_fpr(y_test, dro.predict_proba(X_test), g_test) * 100
    dro_oe   = (1 - dro_metrics["overall_accuracy"]) * 100
    results_list.append({
        "Method": "Group DRO", "MGE": dro_mge, "OA": dro_oa,
        "MFPR": dro_mfpr, "OE": dro_oe
    })
    print_metrics(dro_metrics, "COMPAS", "Group DRO")

    # ---- Method 4: PRIM (Full) ----
    print("\nTraining PRIM (Full)...")
    prim_model = build_model("linear", d, task="classification")
    prim = PRIM(
        model=prim_model,
        G=G,
        epsilon=1.0,
        delta=1e-5,
        rho=0.1,
        lambda_reg=0.01,
        T=100,
        lr=0.01,
        batch_size=64,
        max_grad_norm=1.0,
        alpha=1.2,
        n_pgd_steps=10,
        use_dp=True,
        use_robust=True,
        use_reg=True,
        task="classification",
        device=device,
        verbose=True,
        log_every=20,
    )
    prim.fit(X_train, y_train, g_train)
    prim_metrics = prim.evaluate(X_test, y_test, g_test)
    prim_mge  = prim_metrics["worst_group_error"] * 100
    prim_oa   = prim_metrics["overall_accuracy"] * 100
    prim_mfpr = maximum_fpr(y_test, prim.predict_proba(X_test), g_test) * 100
    prim_oe   = (1 - prim_metrics["overall_accuracy"]) * 100
    results_list.append({
        "Method": "PRIM (Full)", "MGE": prim_mge, "OA": prim_oa,
        "MFPR": prim_mfpr, "OE": prim_oe
    })
    print_metrics(prim_metrics, "COMPAS", "PRIM (Full)")

    # Build and print table
    df = pd.DataFrame(results_list)
    df = df.round(1)
    print("\nTable I: Fairness–Accuracy Trade-off on COMPAS")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "table1_compas_fairness.csv"), index=False)
    print(f"\nSaved to results/table1_compas_fairness.csv")

    return df, prim  # Return PRIM for further experiments


def run_figure_1_pareto_frontier(dataset: dict, device: str = "cpu") -> None:
    """
    Reproduce Figure 1: Fairness–Error Pareto Frontier on COMPAS.

    Sweeps over different lambda_reg values to trace the Pareto frontier
    of overall error vs. worst-group FPR for PRIM and EO-Fair.
    """
    print("\n" + "="*60)
    print("  Figure 1: Fairness–Error Pareto Frontier on COMPAS")
    print("="*60)

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    g_train = dataset["g_train"]
    X_test  = dataset["X_test"]
    y_test  = dataset["y_test"]
    g_test  = dataset["g_test"]
    G       = dataset["n_groups"]
    d       = dataset["input_dim"]

    # Sweep alpha (group boost factor) to trace Pareto frontier
    # Higher alpha → more emphasis on worst group → lower worst-group error but higher overall error
    alpha_values = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5]

    prim_overall_errors = []
    prim_max_fprs = []

    print("\nSweeping PRIM α values for Pareto frontier...")
    for alpha in alpha_values:
        print(f"  α={alpha}...", end=" ", flush=True)
        model = build_model("linear", d, task="classification")
        prim = PRIM(
            model=model, G=G, epsilon=1.0, delta=1e-5, rho=0.1,
            lambda_reg=0.01, T=60, lr=0.01, batch_size=64,
            max_grad_norm=1.0, alpha=alpha, n_pgd_steps=5,
            use_dp=True, use_robust=True, use_reg=True,
            task="classification", device=device, verbose=False, log_every=100,
        )
        prim.fit(X_train, y_train, g_train)
        prim_proba = prim.predict_proba(X_test)

        overall_err = 1.0 - (prim.predict(X_test) == y_test).mean()
        max_fpr = maximum_fpr(y_test, prim_proba, g_test)

        prim_overall_errors.append(overall_err * 100)
        prim_max_fprs.append(max_fpr * 100)
        print(f"OE={overall_err*100:.1f}%, MFPR={max_fpr*100:.1f}%")

    # EO baseline: sweep thresholds
    eo = EqualizedOddsPostProcessing()
    eo.fit(X_train, y_train, g_train)
    eo_proba = eo.predict_proba(X_test)

    eo_overall_errors = []
    eo_max_fprs = []
    for t in np.linspace(0.1, 0.9, 15):
        preds = (eo_proba > t).astype(int)
        oe = (preds != y_test).mean() * 100
        mfpr = maximum_fpr(y_test, preds.astype(float), g_test) * 100
        eo_overall_errors.append(oe)
        eo_max_fprs.append(mfpr)

    # ERM point (no fairness constraint)
    erm_model = build_model("linear", d, task="classification")
    erm = ERM(erm_model, lr=0.01, n_epochs=200, device=device, verbose=False)
    erm.fit(X_train, y_train, g_train)
    erm_proba = erm.predict_proba(X_test)
    erm_oe = (erm.predict(X_test) != y_test).mean() * 100
    erm_mfpr = maximum_fpr(y_test, erm_proba, g_test) * 100

    # Plot Figure 1
    fig, ax = plt.subplots(figsize=(8, 6))

    # PRIM Pareto curve (blue)
    # Sort by MFPR for a clean curve
    sorted_prim = sorted(zip(prim_max_fprs, prim_overall_errors))
    prim_mfprs_sorted, prim_oes_sorted = zip(*sorted_prim)
    ax.plot(prim_mfprs_sorted, prim_oes_sorted, "b-o", linewidth=2, markersize=6,
            label="PRIM (Overall Error)", zorder=3)

    # EO baseline curve (orange)
    sorted_eo = sorted(zip(eo_max_fprs, eo_overall_errors))
    eo_mfprs_sorted, eo_oes_sorted = zip(*sorted_eo)
    ax.plot(eo_mfprs_sorted, eo_oes_sorted, "o-", color="orange", linewidth=2, markersize=6,
            label="Equalized Odds Baseline", zorder=3)

    # ERM point (red X)
    ax.plot(erm_mfpr, erm_oe, "rx", markersize=14, markeredgewidth=3,
            label=f"Unconstrained ERM (FPR={erm_mfpr:.1f}%)", zorder=5)

    # Vertical line at 5% FPR constraint
    ax.axvline(x=5.0, color="gray", linestyle="--", alpha=0.7, label="5% FPR Constraint")

    ax.set_xlabel("Maximum Group False Positive Rate (%)", fontsize=13)
    ax.set_ylabel("Overall Error (%)", fontsize=13)
    ax.set_title("Figure 1: COMPAS Fairness–Error Trade-off\n(PRIM vs. Equalized Odds)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(30, 55)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "figure1_compas_pareto.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Figure 1 saved to {fig_path}")


def run_table_III_robustness(dataset: dict, device: str = "cpu") -> pd.DataFrame:
    """
    Reproduce Table III: Robustness to Adversarial Attacks on COMPAS.

    Evaluates Group DRO and PRIM under PGD attack targeting worst group.
    Reports WGE (worst-group error) and OA (overall accuracy) on clean and attacked data.
    """
    print("\n" + "="*60)
    print("  Table III: Robustness to Adversarial Attacks on COMPAS")
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

    # --- Group DRO: Train then evaluate clean + adversarial ---
    print("\nTraining Group DRO for robustness evaluation...")
    dro_model = build_model("linear", d, task="classification")
    dro = GroupDRO(dro_model, G=G, task="classification", lr=0.01, T=100,
                   batch_size=64, device=device, verbose=False, log_every=100)
    dro.fit(X_train, y_train, g_train)

    # Clean evaluation
    dro_metrics_clean = dro.evaluate(X_test, y_test, g_test)
    dro_wge_clean = dro_metrics_clean["worst_group_error"] * 100
    dro_oa_clean  = dro_metrics_clean["overall_accuracy"] * 100

    # Adversarial evaluation (PGD attack)
    print("  Running PGD attack on Group DRO...")
    dro_model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t  = torch.tensor(y_test, dtype=torch.float32).to(device)

    from src.robustness import pgd_attack
    import torch.nn as nn
    loss_fn_attack = nn.BCEWithLogitsLoss()

    x_adv = pgd_attack(
        model=dro_model,
        x=X_test_t,
        y=y_test_t,
        loss_fn=loss_fn_attack,
        rho=0.1,
        num_steps=20,
        random_start=True,
    )
    with torch.no_grad():
        adv_out = dro_model(x_adv)
        adv_proba = torch.sigmoid(adv_out).cpu().numpy()

    dro_metrics_adv = compute_all_metrics(y_test, adv_proba, g_test, "classification", False)
    dro_wge_adv = dro_metrics_adv["worst_group_error"] * 100
    dro_oa_adv  = dro_metrics_adv["overall_accuracy"] * 100

    results_list += [
        {"Method": "Group DRO (clean)",   "WGE": dro_wge_clean, "OA": dro_oa_clean},
        {"Method": "Group DRO (attacked)", "WGE": dro_wge_adv,   "OA": dro_oa_adv},
    ]

    # --- PRIM Full: Train then evaluate clean + adversarial ---
    print("\nTraining PRIM (Full) for robustness evaluation...")
    prim_model = build_model("linear", d, task="classification")
    prim = PRIM(
        model=prim_model, G=G, epsilon=1.0, delta=1e-5, rho=0.1,
        lambda_reg=0.01, T=100, lr=0.01, batch_size=64,
        max_grad_norm=1.0, alpha=1.2, n_pgd_steps=10,
        use_dp=True, use_robust=True, use_reg=True,
        task="classification", device=device, verbose=False, log_every=50,
    )
    prim.fit(X_train, y_train, g_train)

    # Clean evaluation
    prim_metrics_clean = prim.evaluate(X_test, y_test, g_test)
    prim_wge_clean = prim_metrics_clean["worst_group_error"] * 100
    prim_oa_clean  = prim_metrics_clean["overall_accuracy"] * 100

    # Adversarial evaluation
    print("  Running PGD attack on PRIM...")
    prim.model.eval()
    x_adv_prim = pgd_attack(
        model=prim.model,
        x=X_test_t,
        y=y_test_t,
        loss_fn=loss_fn_attack,
        rho=0.1,
        num_steps=20,
        random_start=True,
    )
    with torch.no_grad():
        adv_out_prim = prim.model(x_adv_prim)
        adv_proba_prim = torch.sigmoid(adv_out_prim).cpu().numpy()

    prim_metrics_adv = compute_all_metrics(y_test, adv_proba_prim, g_test, "classification", False)
    prim_wge_adv = prim_metrics_adv["worst_group_error"] * 100
    prim_oa_adv  = prim_metrics_adv["overall_accuracy"] * 100

    results_list += [
        {"Method": "PRIM Full (clean)",   "WGE": prim_wge_clean, "OA": prim_oa_clean},
        {"Method": "PRIM Full (attacked)", "WGE": prim_wge_adv,   "OA": prim_oa_adv},
    ]

    df = pd.DataFrame(results_list).round(1)
    print("\nTable III: Robustness to Adversarial Attacks on COMPAS")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "table3_robustness.csv"), index=False)
    print(f"\nSaved to results/table3_robustness.csv")

    return df


def run_table_IV_interpretability(dataset: dict, device: str = "cpu") -> pd.DataFrame:
    """
    Reproduce Table IV: Model Interpretability and Complexity on COMPAS.

    Trains four model variants and reports:
      - #Features: Number of features with non-negligible weights
      - Xscore: Interpretability score (%)
      - OA: Overall Accuracy (%)
    """
    print("\n" + "="*60)
    print("  Table IV: Model Interpretability and Complexity on COMPAS")
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

    # 1. Unconstrained Logistic (ERM, linear, no regularization)
    print("\n[1] Unconstrained Logistic Regression...")
    unc_model = build_model("linear", d, task="classification")
    erm = ERM(unc_model, lr=0.01, n_epochs=200, device=device, verbose=False)
    erm.fit(X_train, y_train, g_train)
    erm_metrics = erm.evaluate(X_test, y_test, g_test)
    n_active = len(unc_model.get_active_features(threshold=1e-4))
    results_list.append({
        "Model": "Unconstrained Logistic",
        "#Features": n_active,
        "Xscore": int(unc_model.interpretability_score() * 100),
        "OA": round(erm_metrics["overall_accuracy"] * 100, 1),
    })

    # 2. PRIM Linear (sparse logistic with ℓ₁ regularization)
    print("\n[2] PRIM Linear (ℓ₁ regularized logistic)...")
    linear_model = build_model("linear", d, task="classification")
    prim_linear = PRIM(
        model=linear_model, G=G, epsilon=1.0, delta=1e-5, rho=0.1,
        lambda_reg=0.05,  # Higher λ for more sparsity
        T=100, lr=0.01, batch_size=64, max_grad_norm=1.0, alpha=1.2,
        n_pgd_steps=10, use_dp=True, use_robust=True, use_reg=True,
        task="classification", device=device, verbose=False, log_every=100,
    )
    prim_linear.fit(X_train, y_train, g_train)
    linear_metrics = prim_linear.evaluate(X_test, y_test, g_test)
    n_active_linear = prim_linear.get_active_feature_count(threshold=1e-3)
    results_list.append({
        "Model": "PRIM Linear",
        "#Features": n_active_linear,
        "Xscore": int(prim_linear.get_interpretability_score() * 100),
        "OA": round(linear_metrics["overall_accuracy"] * 100, 1),
    })

    # 3. PRIM Neural (attention-based)
    print("\n[3] PRIM Neural (attention mechanism)...")
    neural_model = build_model("neural", d, hidden_dim=64, task="classification")
    prim_neural = PRIM(
        model=neural_model, G=G, epsilon=1.0, delta=1e-5, rho=0.1,
        lambda_reg=0.01, T=80, lr=0.005, batch_size=64, max_grad_norm=1.0,
        alpha=1.2, n_pgd_steps=8, use_dp=True, use_robust=True, use_reg=True,
        task="classification", device=device, verbose=False, log_every=100,
    )
    prim_neural.fit(X_train, y_train, g_train)
    neural_metrics = prim_neural.evaluate(X_test, y_test, g_test)
    n_active_neural = prim_neural.get_active_feature_count(threshold=0.1)
    results_list.append({
        "Model": "PRIM Neural (attention)",
        "#Features": n_active_neural,
        "Xscore": int(prim_neural.get_interpretability_score() * 100),
        "OA": round(neural_metrics["overall_accuracy"] * 100, 1),
    })

    # 4. PRIM Mixture-of-Experts
    print("\n[4] PRIM Mixture-of-Experts (FairMOE-style)...")
    moe_model = build_model("moe", d, hidden_dim=32, task="classification")
    prim_moe = PRIM(
        model=moe_model, G=G, epsilon=1.0, delta=1e-5, rho=0.1,
        lambda_reg=0.01, T=80, lr=0.005, batch_size=64, max_grad_norm=1.0,
        alpha=1.2, n_pgd_steps=8, use_dp=True, use_robust=True, use_reg=True,
        task="classification", device=device, verbose=False, log_every=100,
    )
    prim_moe.fit(X_train, y_train, g_train)
    moe_metrics = prim_moe.evaluate(X_test, y_test, g_test)
    n_active_moe = prim_moe.get_active_feature_count(threshold=1e-4)
    results_list.append({
        "Model": "PRIM Mixture-of-Experts",
        "#Features": n_active_moe,
        "Xscore": int(prim_moe.get_interpretability_score() * 100),
        "OA": round(moe_metrics["overall_accuracy"] * 100, 1),
    })

    df = pd.DataFrame(results_list)
    print("\nTable IV: Model Interpretability and Complexity on COMPAS")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "table4_interpretability.csv"), index=False)
    print(f"\nSaved to results/table4_interpretability.csv")

    return df


def plot_training_history(prim: PRIM, save_path: str) -> None:
    """
    Plot training history: worst-group error over iterations.
    Shows convergence of PRIM's minimax optimization.
    """
    history = prim.history
    if not history["iteration"]:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Worst-group error and overall error over iterations
    ax1 = axes[0]
    ax1.plot(history["iteration"], [e * 100 for e in history["worst_group_error"]],
             "r-", label="Worst-Group Error", linewidth=2)
    ax1.plot(history["iteration"], [e * 100 for e in history["overall_error"]],
             "b--", label="Overall Error", linewidth=2)
    ax1.set_xlabel("PRIM Iteration", fontsize=12)
    ax1.set_ylabel("Error (%)", fontsize=12)
    ax1.set_title("PRIM Training Convergence", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Group weights over iterations
    ax2 = axes[1]
    weights_array = np.array(history["group_weights"])  # (T, G)
    for g in range(weights_array.shape[1]):
        ax2.plot(history["iteration"], weights_array[:, g],
                 label=f"Group {g} weight", linewidth=2)
    ax2.set_xlabel("PRIM Iteration", fontsize=12)
    ax2.set_ylabel("Group Weight", fontsize=12)
    ax2.set_title("Group Weight Evolution", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training history saved to {save_path}")


def main(args):
    print("\n" + "="*60)
    print("  PRIM-Fair: COMPAS Recidivism Experiment")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    # Load COMPAS dataset
    print("\nLoading COMPAS dataset...")
    try:
        dataset = load_compas()
    except FileNotFoundError:
        print("  COMPAS not found. Using synthetic data for demonstration.")
        print("  Run 'python data/download_datasets.py' for real data.")
        dataset = generate_synthetic_dataset(
            n_samples=2000, n_features=8, G=2, task="classification"
        )
        dataset["name"] = "COMPAS (Synthetic)"

    print(f"  Dataset: {dataset['name']}")
    print(f"  Train: {dataset['n_train']}, Test: {dataset['n_test']}")
    print(f"  Features: {dataset['input_dim']}, Groups: {dataset['n_groups']}")

    # Run requested experiments
    run_all = args.full or (not args.plot and not args.robustness and not args.interpretability)

    prim = None

    if run_all or True:  # Always run Table I as the core result
        df_table1, prim = run_table_I_fairness_accuracy(dataset, device)

    if run_all or args.plot:
        run_figure_1_pareto_frontier(dataset, device)

    if run_all or args.robustness:
        run_table_III_robustness(dataset, device)

    if run_all or args.interpretability:
        run_table_IV_interpretability(dataset, device)

    # Plot training history if PRIM was trained
    if prim is not None:
        plot_training_history(prim, os.path.join(FIGURES_DIR, "compas_training_history.png"))

    print("\n" + "="*60)
    print("  COMPAS experiments complete!")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRIM COMPAS Experiment")
    parser.add_argument("--plot", action="store_true", help="Generate Figure 1 (Pareto frontier)")
    parser.add_argument("--robustness", action="store_true", help="Run Table III (robustness)")
    parser.add_argument("--interpretability", action="store_true", help="Run Table IV (interpretability)")
    parser.add_argument("--full", action="store_true", help="Run all experiments")
    args = parser.parse_args()
    main(args)
