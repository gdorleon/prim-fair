# PRIM-Fair: Private, Robust, Interpretable Minimax Error Fairness

PRIM Experiment
"PRIM-Fair – Private, Robust, Interpretable Minimax Error Fairness"

## Overview

PRIM is a unified machine-learning framework that simultaneously optimises for:
- **Fairness**: Minimax worst-group error (two-player game)
- **Privacy**: (ε, δ)-Differential Privacy via DP-SGD + noisy group-loss queries
- **Robustness**: Adversarial training with PGD inner maximisation (ℓ∞ ball)
- **Interpretability**: ℓ₁ sparsity regulariser + feature-attention neural nets

## Repository Structure
```
prim-fair/
├── src/
│   ├── models.py          # Linear, neural, mixture-of-experts models
│   ├── prim.py            # Core PRIM algorithm (Algorithm 1
│   ├── dp_utils.py        # Differential-privacy helpers (DP-SGD,)
│   ├── robustness.py      # PGD adversarial attack & robust loss
│   ├── datasets.py        # Loaders for all 5 benchmark datasets
│   ├── metrics.py         # Fairness & accuracy metrics
│   └── baselines.py       # ERM, EO-Fair, Group DRO baselines
├── experiments/
│   ├── run_compas.py
│   ├── run_communities.py
│   ├── run_bike.py
│   ├── run_marketing.py
│   ├── run_kdd.py
│   └── run_all.py
├── plots/
│   └── plot_results.py
├── tests/
│   └── test_prim.py
├── data/
├── results/
├── requirements.txt
└── README.md
```
## Quickstart

    git clone https://github.com/paper_under_review/prim-fair.git
    cd prim-fair
    pip install -r requirements.txt
    python experiments/run_compas.py        # single dataset
    python experiments/run_all.py           # all 5 datasets
    python plots/plot_results.py            # reproduce figures

## Hyperparameters

| Parameter  | Default | Description                            |
|------------|---------|----------------------------------------|
| epsilon    | 1.0     | DP privacy budget ε                    |
| delta      | 1e-5    | DP failure probability δ               |
| rho        | 0.1     | Adversarial perturbation radius (ℓ∞)   |
| lambda_reg | 0.01    | Interpretability regulariser weight    |
| T          | 50      | Number of outer minimax iterations     |
| alpha      | 1.5     | Group-weight amplification factor      |
| lr         | 0.01    | Learning rate for model parameters     |
| clip_norm  | 1.0     | DP-SGD gradient clipping norm          |
