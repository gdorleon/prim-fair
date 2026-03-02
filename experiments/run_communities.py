"""
experiments/run_communities.py — Communities and Crime Regression Experiment

Evaluates PRIM on the regression task of predicting violent crime rates.
Protected group: racial composition of community (majority-white vs. minority-majority).

Paper reports: "15% MAE improvement on the worst-off group, while improving overall R²."
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd

from src.datasets import load_communities_crime, generate_synthetic_dataset
from src.models import build_model
from src.prim import PRIM
from src.fairness_metrics import per_group_errors
from sklearn.metrics import mean_absolute_error, r2_score
from baselines.erm import ERM
from baselines.group_dro import GroupDRO

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_communities_experiment(device: str = "cpu") -> pd.DataFrame:
    print("\n" + "="*60)
    print("  Communities and Crime Regression Experiment")
    print("="*60)

    try:
        dataset = load_communities_crime()
    except FileNotFoundError:
        print("  Using synthetic regression data.")
        dataset = generate_synthetic_dataset(
            n_samples=2000, n_features=50, G=2, task="regression"
        )

    X_train, y_train, g_train = dataset["X_train"], dataset["y_train"], dataset["g_train"]
    X_test,  y_test,  g_test  = dataset["X_test"],  dataset["y_test"],  dataset["g_test"]
    G, d = dataset["n_groups"], dataset["input_dim"]

    results_list = []

    def eval_regression(y_true, y_pred, g, method_name):
        """Helper to compute regression metrics."""
        overall_mae = mean_absolute_error(y_true, y_pred)
        try:
            overall_r2 = r2_score(y_true, y_pred)
        except Exception:
            overall_r2 = float("nan")

        group_maes = {}
        for grp in np.unique(g):
            mask = (g == grp)
            if mask.sum() > 0:
                group_maes[grp] = mean_absolute_error(y_true[mask], y_pred[mask])

        worst_mae = max(group_maes.values())
        print(f"\n  {method_name}:")
        print(f"    Overall MAE: {overall_mae:.4f}, R²: {overall_r2:.4f}")
        for grp, mae in sorted(group_maes.items()):
            print(f"    Group {grp} ({dataset['group_names'].get(grp, grp)}): MAE={mae:.4f}")

        results_list.append({
            "Method": method_name,
            "Overall MAE": round(overall_mae, 4),
            "R²": round(overall_r2, 4),
            "Worst-Group MAE": round(worst_mae, 4),
        })

    # ERM
    print("\nTraining ERM...")
    erm_model = build_model("linear", d, output_dim=1, task="regression")
    erm = ERM(erm_model, task="regression", lr=0.01, n_epochs=200, batch_size=64, device=device)
    erm.fit(X_train, y_train, g_train)
    erm_pred = erm.predict(X_test)
    eval_regression(y_test, erm_pred, g_test, "ERM")

    # Group DRO
    print("\nTraining Group DRO...")
    dro_model = build_model("linear", d, output_dim=1, task="regression")
    dro = GroupDRO(dro_model, G=G, task="regression", lr=0.01, T=100,
                   batch_size=64, device=device, verbose=False, log_every=100)
    dro.fit(X_train, y_train, g_train)
    dro_pred = dro.predict(X_test)
    eval_regression(y_test, dro_pred, g_test, "Group DRO")

    # PRIM (no robust, rho=0 as stated in paper)
    print("\nTraining PRIM (Full, ρ=0 as per paper)...")
    prim_model = build_model("linear", d, output_dim=1, task="regression")
    prim = PRIM(
        model=prim_model, G=G, epsilon=1.0, delta=1e-5,
        rho=0.0,  # ρ=0 for Communities (no robustness, as in paper)
        lambda_reg=0.01, T=100, lr=0.01, batch_size=64,
        max_grad_norm=1.0, alpha=1.2, n_pgd_steps=0,
        use_dp=True, use_robust=False, use_reg=True,
        task="regression", device=device, verbose=True, log_every=25,
    )
    prim.fit(X_train, y_train, g_train)
    prim_pred = prim.predict_proba(X_test)
    eval_regression(y_test, prim_pred, g_test, "PRIM (Full)")

    df = pd.DataFrame(results_list)
    print("\nCommunities & Crime — Full Results:")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "communities_results.csv"), index=False)
    return df


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_communities_experiment(device)
