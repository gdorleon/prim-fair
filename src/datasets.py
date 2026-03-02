"""
datasets.py — Dataset Loaders for PRIM Experiments

Loads and preprocesses all five benchmark datasets from the paper:
  1. COMPAS          — Recidivism prediction (binary classification)
  2. Communities & Crime — Violent crime rate (regression)
  3. Seoul Bike Sharing  — Bike rental count (regression)
  4. Bank Marketing      — Term deposit subscription (binary classification)
  5. KDD Cup 99          — Network intrusion detection (binary classification)

Each loader returns a dict with:
  X_train, X_test  — Features (numpy arrays, normalized)
  y_train, y_test  — Labels (numpy arrays)
  g_train, g_test  — Group membership (numpy arrays, int)
  feature_names    — List of feature names
  group_names      — Dict mapping group_id → name
  n_groups         — Number of groups G
  task             — "classification" or "regression"
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Default data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _normalize_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit StandardScaler on training data and transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def _make_dataset_dict(
    X_train, y_train, g_train, X_test, y_test, g_test,
    feature_names, group_names, task, name
) -> dict:
    """Package everything into a consistent dictionary."""
    return {
        "name": name,
        "X_train": X_train.astype(np.float32),
        "y_train": y_train,
        "g_train": g_train.astype(int),
        "X_test": X_test.astype(np.float32),
        "y_test": y_test,
        "g_test": g_test.astype(int),
        "feature_names": feature_names,
        "group_names": group_names,
        "n_groups": len(group_names),
        "task": task,
        "input_dim": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ---------------------------------------------------------------------------
# 1. COMPAS Dataset
# ---------------------------------------------------------------------------

def load_compas(
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    use_cross_validation: bool = False,
    random_state: int = 42,
) -> dict:
    """
    Load and preprocess the COMPAS recidivism dataset.

    Protected attribute: Race (Caucasian vs. African-American)
    Task: Binary classification (will reoffend within 2 years?)

    Source: ProPublica's COMPAS analysis
    URL: https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv

    Features used (consistent with paper's 9 features):
        age, priors_count, days_b_screening_arrest, juv_fel_count,
        juv_misd_count, juv_other_count, c_charge_degree, sex, race (excluded from features)

    Groups:
        0 = Caucasian
        1 = African-American
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR, "compas-scores-two-years.csv")

    print(f"Loading COMPAS from {data_path}...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"COMPAS file not found at {data_path}.\n"
            "Run: python data/download_datasets.py"
        )

    df = pd.read_csv(data_path)

    # Standard ProPublica filtering (as used in fairness literature)
    df = df[
        (df["days_b_screening_arrest"] <= 30) &
        (df["days_b_screening_arrest"] >= -30) &
        (df["is_recid"] != -1) &
        (df["c_charge_degree"] != "O") &
        (df["score_text"] != "N/A")
    ].copy()

    # Keep only Black and White defendants (standard fairness comparison)
    df = df[df["race"].isin(["African-American", "Caucasian"])].copy()

    # Encode features
    df["sex_binary"] = (df["sex"] == "Male").astype(int)
    df["charge_degree_binary"] = (df["c_charge_degree"] == "F").astype(int)

    # The 9 features mentioned in the paper
    feature_cols = [
        "age",
        "priors_count",
        "days_b_screening_arrest",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "charge_degree_binary",
        "sex_binary",
    ]
    feature_names = feature_cols  # 8 features (race excluded per paper)

    X = df[feature_cols].fillna(0).values
    y = df["two_year_recid"].values.astype(int)

    # Group: 0 = Caucasian, 1 = African-American
    g = (df["race"] == "African-American").astype(int).values
    group_names = {0: "Caucasian", 1: "African-American"}

    # Train/test split (80/20 as stated in paper, or 5-fold CV for smaller sets)
    if use_cross_validation:
        # For cross-validation experiments, return full data with CV indices
        return {
            "name": "COMPAS",
            "X": X.astype(np.float32),
            "y": y,
            "g": g.astype(int),
            "feature_names": feature_names,
            "group_names": group_names,
            "n_groups": 2,
            "task": "classification",
            "input_dim": X.shape[1],
            "use_cv": True,
        }

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, g, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test, _ = _normalize_features(X_train, X_test)

    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Groups: {dict(zip(*np.unique(g_train, return_counts=True)))}")

    return _make_dataset_dict(
        X_train, y_train, g_train, X_test, y_test, g_test,
        feature_names, group_names, "classification", "COMPAS"
    )


# ---------------------------------------------------------------------------
# 2. Communities and Crime Dataset
# ---------------------------------------------------------------------------

def load_communities_crime(
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Load and preprocess the Communities and Crime dataset.

    Protected attribute: Racial majority (>= 50% Black → minority group)
    Task: Regression — predict violent crimes per 100k population

    Source: UCI Machine Learning Repository
    URL: https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data

    Groups:
        0 = Majority-white community
        1 = Minority-majority community (>50% Black population)
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR, "communities.data")

    print(f"Loading Communities & Crime from {data_path}...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Communities data not found at {data_path}.\n"
            "Run: python data/download_datasets.py"
        )

    # Dataset has 128 columns; first 5 are non-predictive (state, county, etc.)
    # Column 5-127 are features; last column (127) is target (ViolentCrimesPerPop)
    df = pd.read_csv(data_path, header=None, na_values="?")

    # Drop the first 5 non-predictive columns (identifiers)
    # Column 3 is racepctblack (fraction Black) — used for group definition
    race_col = df.iloc[:, 3].copy()

    # Drop rows with missing target
    target_col = df.iloc[:, -1]
    valid_mask = target_col.notna()
    df = df[valid_mask].copy()
    race_col = race_col[valid_mask]

    # Remove first 5 identifier columns and use remaining as features
    X_raw = df.iloc[:, 5:].copy()

    # Replace remaining missing values with column median
    X_raw = X_raw.fillna(X_raw.median())

    # Target: violent crimes per population (last column)
    y = df.iloc[:, -1].values.astype(np.float32)

    # Features: all except target
    X = X_raw.iloc[:, :-1].values.astype(np.float32)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Group: minority community = racepctblack >= 0.5
    race_pct = race_col.fillna(race_col.median()).values
    g = (race_pct >= 0.5).astype(int)
    group_names = {0: "Majority-White", 1: "Minority-Majority"}

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, g, test_size=test_size, random_state=random_state
    )
    X_train, X_test, _ = _normalize_features(X_train, X_test)

    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Groups: {dict(zip(*np.unique(g_train, return_counts=True)))}")

    return _make_dataset_dict(
        X_train, y_train, g_train, X_test, y_test, g_test,
        feature_names, group_names, "regression", "Communities & Crime"
    )


# ---------------------------------------------------------------------------
# 3. Seoul Bike Sharing Dataset
# ---------------------------------------------------------------------------

def load_bike_sharing(
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Load and preprocess the Seoul Bike Sharing dataset.

    Protected attribute: Season (Summer [3] vs. Winter [1])
    Task: Regression — predict hourly bike rental count

    Source: UCI Machine Learning Repository
    URL: https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand

    Groups:
        0 = Summer (highest rental season)
        1 = Winter (lowest rental season)
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR, "SeoulBikeData.csv")

    print(f"Loading Seoul Bike Sharing from {data_path}...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Seoul Bike data not found at {data_path}.\n"
            "Run: python data/download_datasets.py"
        )

    df = pd.read_csv(data_path, encoding="unicode_escape")

    # Drop date column (not useful as raw string)
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    # Rename for clarity (column order from dataset docs)
    # Columns: Rented Bike Count, Hour, Temperature(C), Humidity(%), Wind speed(m/s),
    #          Visibility(10m), Dew point temperature(C), Solar Radiation(MJ/m2),
    #          Rainfall(mm), Snowfall(cm), Seasons, Holiday, Functioning Day
    df.columns = [c.strip() for c in df.columns]

    # Encode categorical columns
    if "Holiday" in df.columns:
        df["Holiday"] = (df["Holiday"] == "Holiday").astype(int)
    if "Functioning Day" in df.columns:
        df["Functioning Day"] = (df["Functioning Day"] == "Yes").astype(int)

    # Keep only summer (3) and winter (1) seasons as the two groups
    seasons_col = "Seasons"
    if seasons_col in df.columns:
        df = df[df[seasons_col].isin([1, 3])].copy()
        # Group: 0=Summer, 1=Winter
        g = (df[seasons_col] == 1).astype(int).values  # 1=Winter → group 1
    else:
        # Try numeric season column
        g = np.zeros(len(df), dtype=int)

    group_names = {0: "Summer", 1: "Winter"}

    # Target: bike rental count
    target_col = [c for c in df.columns if "Rented" in c or "Count" in c][0]
    y = df[target_col].values.astype(np.float32)

    # Features: all except target and season (season used for grouping)
    feature_cols = [c for c in df.columns if c not in [target_col, seasons_col]]
    X = df[feature_cols].values.astype(np.float32)
    feature_names = feature_cols

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, g, test_size=test_size, random_state=random_state
    )
    X_train, X_test, _ = _normalize_features(X_train, X_test)

    # Normalize y for regression (scale to [0, 1] range approximately)
    y_scale = y_train.max() + 1e-8
    y_train = y_train / y_scale
    y_test = y_test / y_scale

    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Groups: Summer vs Winter")

    return _make_dataset_dict(
        X_train, y_train, g_train, X_test, y_test, g_test,
        feature_names, group_names, "regression", "Bike Sharing"
    )


# ---------------------------------------------------------------------------
# 4. Bank Marketing Dataset
# ---------------------------------------------------------------------------

def load_marketing(
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Load and preprocess the Portuguese Bank Marketing dataset.

    Protected attribute: Job type (grouped into manual/blue-collar vs. other)
    Task: Binary classification — will client subscribe to term deposit?

    Source: UCI Machine Learning Repository
    URL: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

    Groups:
        0 = White-collar / professional jobs
        1 = Blue-collar / manual jobs (socio-economically disadvantaged)
    """
    if data_path is None:
        # Try both ; and , separated versions
        for fname in ["bank-additional-full.csv", "bank-full.csv", "bank.csv"]:
            candidate = os.path.join(DATA_DIR, fname)
            if os.path.exists(candidate):
                data_path = candidate
                break

    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Bank Marketing data not found.\n"
            "Run: python data/download_datasets.py"
        )

    print(f"Loading Bank Marketing from {data_path}...")

    # The file uses semicolon separators
    try:
        df = pd.read_csv(data_path, sep=";")
    except Exception:
        df = pd.read_csv(data_path, sep=",")

    # Target: 'y' column — 'yes'/'no' subscription
    y = (df["y"] == "yes").astype(int).values

    # Group: Blue-collar vs. other jobs
    blue_collar_jobs = ["blue-collar", "services", "housemaid", "technician"]
    if "job" in df.columns:
        g = df["job"].apply(lambda x: 1 if x in blue_collar_jobs else 0).values
    else:
        g = np.zeros(len(df), dtype=int)
    group_names = {0: "White-collar", 1: "Blue-collar"}

    # One-hot encode categorical features
    categorical_cols = [c for c in df.columns if df[c].dtype == object and c not in ["y", "job"]]
    feature_cols = [c for c in df.columns if c not in ["y", "job"]]

    # Encode categoricals
    df_features = df[feature_cols].copy()
    for col in categorical_cols:
        if col in df_features.columns:
            dummies = pd.get_dummies(df_features[col], prefix=col, drop_first=True)
            df_features = pd.concat([df_features.drop(columns=[col]), dummies], axis=1)

    # Replace any remaining non-numeric
    df_features = df_features.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df_features.values.astype(np.float32)
    feature_names = df_features.columns.tolist()

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, g, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test, _ = _normalize_features(X_train, X_test)

    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Groups: {dict(zip(*np.unique(g_train, return_counts=True)))}")

    return _make_dataset_dict(
        X_train, y_train, g_train, X_test, y_test, g_test,
        feature_names, group_names, "classification", "Marketing"
    )


# ---------------------------------------------------------------------------
# 5. KDD Cup 99 (Internet Traffic / Intrusion Detection)
# ---------------------------------------------------------------------------

def load_internet_traffic(
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    subsample: int = 50000,
) -> dict:
    """
    Load and preprocess the KDD Cup 99 network intrusion dataset.

    Protected attribute: Protocol type (TCP=0, UDP=1, ICMP=2) → binary: TCP vs non-TCP
    Task: Binary classification — normal vs. attack traffic

    Source: UCI / KDD Archive
    URL: https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data

    We subsample to 50,000 rows for computational feasibility (as in paper).

    Groups:
        0 = TCP protocol
        1 = UDP/ICMP protocol
    """
    if data_path is None:
        for fname in ["kddcup.data_10_percent_corrected", "kddcup.data.gz",
                      "kddcup99.csv", "kdd99.csv"]:
            candidate = os.path.join(DATA_DIR, fname)
            if os.path.exists(candidate):
                data_path = candidate
                break

    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError(
            f"KDD Cup 99 data not found.\n"
            "Run: python data/download_datasets.py"
        )

    print(f"Loading KDD Cup 99 from {data_path}...")

    # Column names for KDD Cup 99
    col_names = [
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label"
    ]

    try:
        df = pd.read_csv(data_path, header=None, names=col_names, nrows=subsample * 3)
    except Exception:
        df = pd.read_csv(data_path, names=col_names, nrows=subsample * 3)

    # Subsample for computational feasibility
    if len(df) > subsample:
        df = df.sample(n=subsample, random_state=random_state).reset_index(drop=True)

    # Binary label: normal vs. attack
    y = (df["label"] != "normal.").astype(int).values

    # Group by protocol type
    protocol_map = {"tcp": 0, "udp": 1, "icmp": 1}
    g = df["protocol_type"].map(protocol_map).fillna(1).astype(int).values
    group_names = {0: "TCP", 1: "UDP/ICMP"}

    # Encode categorical features
    categorical_cols = ["protocol_type", "service", "flag"]
    df_features = df.drop(columns=["label"]).copy()

    for col in categorical_cols:
        if col in df_features.columns:
            dummies = pd.get_dummies(df_features[col], prefix=col, drop_first=True)
            df_features = pd.concat([df_features.drop(columns=[col]), dummies], axis=1)

    df_features = df_features.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = df_features.values.astype(np.float32)
    feature_names = df_features.columns.tolist()

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, g, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test, _ = _normalize_features(X_train, X_test)

    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Groups: TCP vs UDP/ICMP")
    print(f"  Features: {X.shape[1]}")

    return _make_dataset_dict(
        X_train, y_train, g_train, X_test, y_test, g_test,
        feature_names, group_names, "classification", "Internet Traffic (KDD99)"
    )


# ---------------------------------------------------------------------------
# Generate Synthetic Dataset (for quick testing without downloading)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_samples: int = 2000,
    n_features: int = 10,
    G: int = 2,
    task: str = "classification",
    noise: float = 0.1,
    random_state: int = 42,
) -> dict:
    """
    Generate a synthetic dataset for quick unit testing and debugging.

    Creates a dataset where:
      - Group 0 has abundant samples and lower noise
      - Group 1 has fewer samples and higher noise (causing unfairness in ERM)

    This mimics real-world imbalanced group scenarios like COMPAS.

    Args:
        n_samples:    Total number of samples.
        n_features:   Number of features.
        G:            Number of groups.
        task:         "classification" or "regression".
        noise:        Noise level.
        random_state: Random seed.

    Returns:
        Dataset dict with same format as real loaders.
    """
    rng = np.random.RandomState(random_state)

    # True weight vector (what we're trying to learn)
    true_w = rng.randn(n_features)
    true_w[:n_features // 2] = 0  # Half features are irrelevant (sparse ground truth)

    # Assign groups: 70/30 split between group 0 and group 1
    group_probs = [0.7] + [0.3 / (G - 1)] * (G - 1)
    g = rng.choice(G, size=n_samples, p=group_probs)

    # Generate features with group-specific means (creates distribution shift)
    X = rng.randn(n_samples, n_features)
    for grp in range(G):
        mask = (g == grp)
        X[mask] += grp * 0.5  # Group 1 has slightly shifted distribution

    # Generate labels
    logits = X @ true_w + rng.randn(n_samples) * noise

    if task == "classification":
        proba = 1.0 / (1.0 + np.exp(-logits))
        y = (proba > 0.5).astype(int)
    else:
        y = logits.astype(np.float32)

    # Split
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X.astype(np.float32), y, g,
        test_size=0.2, random_state=random_state
    )
    X_train, X_test, _ = _normalize_features(X_train, X_test)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    group_names = {i: f"Group_{i}" for i in range(G)}

    return _make_dataset_dict(
        X_train, y_train, g_train, X_test, y_test, g_test,
        feature_names, group_names, task, "Synthetic"
    )
