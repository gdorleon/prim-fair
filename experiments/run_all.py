"""
experiments/run_all.py — Run All PRIM Experiments

Master script that runs all experiments and generates all tables and figures
from the PRIM-Fair paper:

  Table I:   Fairness–Accuracy Trade-off on COMPAS
  Table II:  Impact of Differential Privacy on Marketing
  Table III: Robustness to Adversarial Attacks on COMPAS
  Table IV:  Model Interpretability and Complexity on COMPAS
  Figure 1:  Pareto Frontier (COMPAS)
  + Additional experiments on Communities, Bike, Internet datasets

Usage:
    python experiments/run_all.py                    # Run all
    python experiments/run_all.py --quick            # Fewer iterations (for testing)
    python experiments/run_all.py --dataset compas   # Run single dataset
"""

import sys
import os
import argparse
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def print_header(title: str, width: int = 60) -> None:
    """Print a formatted section header."""
    print("\n" + "="*width)
    print(f"  {title}")
    print("="*width)


def run_experiment_safe(name: str, fn, *args, **kwargs):
    """Run an experiment function with error handling — failures don't block other experiments."""
    print_header(f"Running: {name}")
    start = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"\n  ✓ {name} completed in {elapsed:.1f}s")
        return result, True
    except KeyboardInterrupt:
        print(f"\n  ⚡ {name} interrupted by user")
        raise
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ✗ {name} failed after {elapsed:.1f}s")
        print(f"    Error: {e}")
        traceback.print_exc()
        return None, False


def main():
    parser = argparse.ArgumentParser(
        description="Run all PRIM experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_all.py                   # Run all experiments
  python experiments/run_all.py --quick           # Quick test run (fewer iterations)
  python experiments/run_all.py --dataset compas  # Run COMPAS only
  python experiments/run_all.py --dataset marketing
  python experiments/run_all.py --no-communities  # Skip slow datasets
        """
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer iterations for testing")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "compas", "communities", "bike", "marketing", "internet"],
                        help="Which dataset to run")
    parser.add_argument("--no-communities", action="store_true",
                        help="Skip Communities & Crime (can be slow)")
    parser.add_argument("--no-internet", action="store_true",
                        help="Skip Internet Traffic (can be slow)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_header("PRIM-Fair: All Experiments")
    print(f"  Device: {device}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Figures dir: {FIGURES_DIR}")

    # Summary of all results
    all_results = {}
    success_flags = {}
    overall_start = time.time()

    # ============================================================
    # COMPAS Experiments (Tables I, III, IV + Figure 1)
    # ============================================================
    if args.dataset in ("all", "compas"):
        from experiments.run_compas import (
            run_table_I_fairness_accuracy,
            run_figure_1_pareto_frontier,
            run_table_III_robustness,
            run_table_IV_interpretability,
        )
        from src.datasets import load_compas, generate_synthetic_dataset

        print_header("Loading COMPAS Dataset")
        try:
            dataset = load_compas()
        except FileNotFoundError:
            print("  COMPAS not found. Using synthetic data (run data/download_datasets.py for real data).")
            dataset = generate_synthetic_dataset(n_samples=2000, n_features=8, G=2, task="classification")

        # Table I
        result, ok = run_experiment_safe(
            "Table I: COMPAS Fairness–Accuracy", run_table_I_fairness_accuracy, dataset, device
        )
        if result:
            df_t1, prim_t1 = result
            all_results["Table I"] = df_t1
            success_flags["Table I"] = ok

        # Figure 1 (Pareto frontier)
        if not args.quick:
            _, ok = run_experiment_safe(
                "Figure 1: COMPAS Pareto Frontier", run_figure_1_pareto_frontier, dataset, device
            )
            success_flags["Figure 1"] = ok

        # Table III (Robustness)
        result, ok = run_experiment_safe(
            "Table III: COMPAS Robustness", run_table_III_robustness, dataset, device
        )
        if result is not None:
            all_results["Table III"] = result
        success_flags["Table III"] = ok

        # Table IV (Interpretability)
        result, ok = run_experiment_safe(
            "Table IV: COMPAS Interpretability", run_table_IV_interpretability, dataset, device
        )
        if result is not None:
            all_results["Table IV"] = result
        success_flags["Table IV"] = ok

    # ============================================================
    # Marketing Experiment (Table II: DP Impact)
    # ============================================================
    if args.dataset in ("all", "marketing"):
        from experiments.run_marketing import run_dp_impact_experiment, run_epsilon_sweep
        from src.datasets import load_marketing, generate_synthetic_dataset

        print_header("Loading Marketing Dataset")
        try:
            dataset = load_marketing()
        except FileNotFoundError:
            print("  Using synthetic data.")
            dataset = generate_synthetic_dataset(n_samples=5000, n_features=30, G=2, task="classification")

        result, ok = run_experiment_safe(
            "Table II: Marketing DP Impact", run_dp_impact_experiment, dataset, device
        )
        if result:
            df_t2, _ = result
            all_results["Table II"] = df_t2
        success_flags["Table II"] = ok

        if not args.quick:
            _, ok = run_experiment_safe(
                "Marketing ε Sweep", run_epsilon_sweep, dataset, device
            )
            success_flags["Marketing ε Sweep"] = ok

    # ============================================================
    # Communities & Crime (Regression)
    # ============================================================
    if args.dataset in ("all", "communities") and not args.no_communities:
        from experiments.run_communities import run_communities_experiment
        result, ok = run_experiment_safe(
            "Communities & Crime Regression", run_communities_experiment, device
        )
        if result is not None:
            all_results["Communities"] = result
        success_flags["Communities"] = ok

    # ============================================================
    # Bike Sharing (Regression)
    # ============================================================
    if args.dataset in ("all", "bike"):
        from experiments.run_bike import run_bike_experiment
        result, ok = run_experiment_safe(
            "Seoul Bike Sharing Regression", run_bike_experiment, device
        )
        if result is not None:
            all_results["Bike"] = result
        success_flags["Bike"] = ok

    # ============================================================
    # Internet Traffic / KDD99 (Classification)
    # ============================================================
    if args.dataset in ("all", "internet") and not args.no_internet:
        from experiments.run_internet import run_internet_experiment
        result, ok = run_experiment_safe(
            "KDD Cup 99 Intrusion Detection", run_internet_experiment, device
        )
        if result is not None:
            all_results["Internet"] = result
        success_flags["Internet"] = ok

    # ============================================================
    # Final Summary
    # ============================================================
    total_time = time.time() - overall_start
    print_header("Experiment Summary")
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n  Status:")
    for name, ok in success_flags.items():
        status = "✓" if ok else "✗"
        print(f"    {status} {name}")

    print(f"\n  Generated tables:")
    for name, df in all_results.items():
        path = os.path.join(RESULTS_DIR, f"{name.lower().replace(' ', '_')}_summary.csv")
        df.to_csv(path, index=False)
        print(f"    → {path}")

    print(f"\n  All figures saved to: {FIGURES_DIR}")
    print(f"  All results saved to: {RESULTS_DIR}")

    # Print all tables
    for name, df in all_results.items():
        print(f"\n  {name}:")
        print(df.to_string(index=False))

    print_header("All PRIM Experiments Complete!")


if __name__ == "__main__":
    main()
