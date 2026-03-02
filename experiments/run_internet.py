"""
experiments/run_internet.py — KDD Cup 99 Network Intrusion Experiment

Protected group: Protocol type (TCP vs UDP/ICMP).
Paper uses ε=2 (relaxed DP for convergence on this dataset).
Dataset is subsampled to 50k samples.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd

from src.datasets import load_internet_traffic, generate_synthetic_dataset
from src.models import build_model
from src.prim import PRIM
from src.fairness_metrics import compute_all_metrics, print_metrics
from baselines.erm import ERM
from baselines.group_dro import GroupDRO

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_internet_experiment(device: str = "cpu") -> pd.DataFrame:
    print("\n" + "="*60)
    print("  KDD Cup 99 Internet Intrusion Detection Experiment")
    print("="*60)
    print("  Note: ε=2 as in paper (relaxed budget for convergence on this dataset)")

    try:
        dataset = load_internet_traffic(subsample=50000)
    except FileNotFoundError:
        print("  Using synthetic network data.")
        dataset = generate_synthetic_dataset(
            n_samples=10000, n_features=40, G=2, task="classification"
        )

    X_train, y_train, g_train = dataset["X_train"], dataset["y_train"], dataset["g_train"]
    X_test,  y_test,  g_test  = dataset["X_test"],  dataset["y_test"],  dataset["g_test"]
    G, d = dataset["n_groups"], dataset["input_dim"]

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {d}")

    results_list = []

    # ERM
    print("\nTraining ERM...")
    erm_model = build_model("linear", d, task="classification")
    erm = ERM(erm_model, lr=0.01, n_epochs=50, batch_size=256, device=device, verbose=False)
    erm.fit(X_train, y_train, g_train)
    erm_metrics = erm.evaluate(X_test, y_test, g_test)
    print_metrics(erm_metrics, "KDD99", "ERM")
    results_list.append({
        "Method": "ERM",
        "Worst-Group Error (%)": round(erm_metrics["worst_group_error"] * 100, 2),
        "Overall Accuracy (%)": round(erm_metrics["overall_accuracy"] * 100, 2),
    })

    # Group DRO
    print("\nTraining Group DRO...")
    dro_model = build_model("linear", d, task="classification")
    dro = GroupDRO(dro_model, G=G, task="classification", lr=0.01, T=50,
                   batch_size=256, device=device, verbose=False, log_every=100)
    dro.fit(X_train, y_train, g_train)
    dro_metrics = dro.evaluate(X_test, y_test, g_test)
    print_metrics(dro_metrics, "KDD99", "Group DRO")
    results_list.append({
        "Method": "Group DRO",
        "Worst-Group Error (%)": round(dro_metrics["worst_group_error"] * 100, 2),
        "Overall Accuracy (%)": round(dro_metrics["overall_accuracy"] * 100, 2),
    })

    # PRIM Full (ε=2 as in paper, ρ=0.05)
    print("\nTraining PRIM Full (ε=2)...")
    prim_model = build_model("linear", d, task="classification")
    prim = PRIM(
        model=prim_model, G=G,
        epsilon=2.0,  # Paper uses ε=2 for KDD99 for convergence
        delta=1e-5, rho=0.05,
        lambda_reg=0.01, T=50, lr=0.01, batch_size=256,
        max_grad_norm=1.0, alpha=1.2, n_pgd_steps=5,
        use_dp=True, use_robust=True, use_reg=True,
        task="classification", device=device, verbose=True, log_every=10,
    )
    prim.fit(X_train, y_train, g_train)
    prim_metrics = prim.evaluate(X_test, y_test, g_test)
    print_metrics(prim_metrics, "KDD99", "PRIM Full")
    results_list.append({
        "Method": "PRIM Full (ε=2)",
        "Worst-Group Error (%)": round(prim_metrics["worst_group_error"] * 100, 2),
        "Overall Accuracy (%)": round(prim_metrics["overall_accuracy"] * 100, 2),
    })

    df = pd.DataFrame(results_list)
    print("\nKDD Cup 99 Results:")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "internet_kdd99_results.csv"), index=False)
    return df


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_internet_experiment(device)
