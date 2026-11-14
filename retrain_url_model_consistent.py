
import json
import re
import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from backend.app.services.url_feature_extractor import (
    extract_url_features,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "url_clean.csv"
MODEL_PATH = ROOT / "models" / "url_model_extracted.joblib"
REPORT_PATH = ROOT / "reports" / "url_extractor_model_evaluation.csv"
METRICS_PATH = ROOT / "reports" / "url_extractor_model_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_COLUMNS = [
    "url_length",
    "has_ip_address",
    "dot_count",
    "https_flag",
    "url_entropy",
    "token_count",
    "subdomain_count",
    "query_param_count",
    "tld_length",
    "path_length",
    "has_hyphen_in_domain",
    "number_of_digits",
    "tld_popularity",
    "suspicious_file_extension",
    "domain_name_length",
    "percentage_numeric_chars",
]


def find_column(dataframe, candidates):
    lookup = {column.lower(): column for column in dataframe.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not find any of these columns: {candidates}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def get_hostname(url):
    value = str(url).strip()

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        value = "http://" + value

    hostname = urlsplit(value).hostname

    if not hostname:
        raise ValueError(f"Invalid URL: {url}")

    hostname = hostname.lower().rstrip(".")

    try:
        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        return hostname.removeprefix("www.")


def main():
    print("Loading processed URL dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    dataframe = pd.read_csv(DATA_PATH)

    url_column = find_column(dataframe, ["url", "URL", "website"])
    target_column = find_column(dataframe, ["target", "label", "classlabel"])

    dataframe = dataframe[[url_column, target_column]].copy()
    dataframe = dataframe.dropna(subset=[url_column, target_column])

    dataframe[target_column] = pd.to_numeric(
        dataframe[target_column], errors="coerce"
    )
    dataframe = dataframe.dropna(subset=[target_column])
    dataframe[target_column] = dataframe[target_column].astype(int)

    if not set(dataframe[target_column].unique()).issubset({0, 1}):
        raise ValueError("Target column must contain only 0 and 1.")

    print(f"Rows available: {len(dataframe):,}")
    print("Target distribution:")
    print(dataframe[target_column].value_counts().sort_index())

    print("\nExtracting features from raw URLs...")

    feature_rows = []
    valid_indices = []
    invalid_count = 0

    for index, url in dataframe[url_column].items():
        try:
            features = extract_url_features(str(url))
            feature_rows.append(features)
            valid_indices.append(index)
        except Exception as error:
            invalid_count += 1
            if invalid_count <= 5:
                print(f"Skipped invalid URL: {url} | {error}")

    if not feature_rows:
        raise ValueError("No valid URLs were available for feature extraction.")

    features_df = pd.DataFrame(
        feature_rows,
        columns=FEATURE_COLUMNS,
        index=valid_indices,
    )

    labels = dataframe.loc[valid_indices, target_column].astype(int)

    features_df = features_df.replace([np.inf, -np.inf], np.nan)

    print(f"Valid URLs: {len(features_df):,}")
    print(f"Invalid URLs skipped: {invalid_count:,}")
    print(f"Feature columns: {len(features_df.columns)}")

    if features_df.isna().all().any():
        raise ValueError("At least one feature column contains only missing values.")

    print("\nCreating hostname-separated train/test split...")

    groups = [
        get_hostname(url)
        for url in dataframe.loc[valid_indices, url_column]
    ]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(features_df, labels, groups=groups)
    )

    X_train = features_df.iloc[train_indices]
    X_test = features_df.iloc[test_indices]
    y_train = labels.iloc[train_indices]
    y_test = labels.iloc[test_indices]

    train_hosts = set(np.array(groups)[train_indices])
    test_hosts = set(np.array(groups)[test_indices])
    overlap = train_hosts.intersection(test_hosts)

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")
    print(f"Training hostnames: {len(train_hosts):,}")
    print(f"Testing hostnames: {len(test_hosts):,}")
    print(f"Hostname overlap: {len(overlap)}")

    if overlap:
        raise RuntimeError("Hostname overlap detected. Stopping training.")

    print("\nTraining Random Forest URL model...")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_hostnames": int(len(train_hosts)),
        "test_hostnames": int(len(test_hosts)),
        "hostname_overlap": int(len(overlap)),
        "invalid_urls_skipped": int(invalid_count),
        "feature_count": len(FEATURE_COLUMNS),
        "target_column": target_column,
        "positive_class_meaning": "Malicious",
    }

    print("\n========== URL MODEL EVALUATION ==========")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    ROOT.joinpath("models").mkdir(exist_ok=True)
    ROOT.joinpath("reports").mkdir(exist_ok=True)

    artifact = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "target_meaning": {
            "0": "Legitimate",
            "1": "Malicious",
        },
        "feature_extractor": "backend.app.services.url_feature_extractor",
        "validation": "hostname_group_split",
    }

    joblib.dump(artifact, MODEL_PATH)

    pd.DataFrame([metrics]).to_csv(REPORT_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Evaluation report saved: {REPORT_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    print("\nURL model retraining completed successfully.")


if __name__ == "__main__":
    main()
