"""
experiments/run_bike.py — Seoul Bike Sharing Regression Experiment

Protected group: Season (Summer vs. Winter bike rentals).
Paper reports: "12% lower error during unexpected weather conditions."
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd

from src.datasets import load_bike_sharing, generate_synthetic_dataset
from src.models import build_model
from src.prim import PRIM
from sklearn.metrics import mean_absolute_error
from baselines.erm import ERM
from baselines.group_dro import GroupDRO

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_bike_experiment(device: str = "cpu") -> pd.DataFrame:
    print("\n" + "="*60)
    print("  Seoul Bike Sharing Regression Experiment")
    print("="*60)

    try:
        dataset = load_bike_sharing()
    except FileNotFoundError:
        print("  Using synthetic data.")
        dataset = generate_synthetic_dataset(
            n_samples=3000, n_features=11, G=2, task="regression"
        )

    X_train, y_train, g_train = dataset["X_train"], dataset["y_train"], dataset["g_train"]
    X_test,  y_test,  g_test  = dataset["X_test"],  dataset["y_test"],  dataset["g_test"]
    G, d = dataset["n_groups"], dataset["input_dim"]

    results_list = []

    def eval_and_record(y_true, y_pred, g, method_name):
        overall_mae = mean_absolute_error(y_true, y_pred)
        group_maes = {}
        for grp in np.unique(g):
            mask = (g == grp)
            if mask.sum() > 0:
                group_maes[grp] = mean_absolute_error(y_true[mask], y_pred[mask])
        worst_mae = max(group_maes.values())
        print(f"  {method_name}: Overall MAE={overall_mae:.4f}, Worst-Group MAE={worst_mae:.4f}")
        results_list.append({
            "Method": method_name, "Overall MAE": round(overall_mae, 4),
            "Worst-Group MAE": round(worst_mae, 4)
        })

    # ERM
    erm_model = build_model("linear", d, output_dim=1, task="regression")
    erm = ERM(erm_model, task="regression", lr=0.01, n_epochs=200, device=device)
    erm.fit(X_train, y_train, g_train)
    eval_and_record(y_test, erm.predict(X_test), g_test, "ERM")

    # Group DRO
    dro_model = build_model("linear", d, output_dim=1, task="regression")
    dro = GroupDRO(dro_model, G=G, task="regression", lr=0.01, T=100,
                   batch_size=64, device=device, verbose=False, log_every=100)
    dro.fit(X_train, y_train, g_train)
    eval_and_record(y_test, dro.predict(X_test), g_test, "Group DRO")

    # PRIM (ρ=0 as in paper for Bike dataset)
    prim_model = build_model("neural", d, hidden_dim=32, output_dim=1, task="regression")
    prim = PRIM(
        model=prim_model, G=G, epsilon=1.0, delta=1e-5, rho=0.0,
        lambda_reg=0.01, T=80, lr=0.005, batch_size=64,
        max_grad_norm=1.0, alpha=1.2, n_pgd_steps=0,
        use_dp=True, use_robust=False, use_reg=True,
        task="regression", device=device, verbose=True, log_every=20,
    )
    prim.fit(X_train, y_train, g_train)
    eval_and_record(y_test, prim.predict_proba(X_test), g_test, "PRIM (Full)")

    df = pd.DataFrame(results_list)
    print("\nBike Sharing Results:")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "bike_results.csv"), index=False)
    return df


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_bike_experiment(device)
