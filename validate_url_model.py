
import os
import re
import joblib
import numpy as np
import pandas as pd

from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

os.makedirs("reports", exist_ok=True)

DATA_PATH = "data/processed/url_clean.csv"

FEATURES = [
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
    "percentage_numeric_chars"
]


def extract_hostname(value):
    value = str(value).strip()

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        value = "http://" + value

    try:
        hostname = urlsplit(value).hostname
        return hostname.lower().removeprefix("www.") if hostname else None
    except ValueError:
        return None


print("Loading URL dataset...")

df = pd.read_csv(DATA_PATH)

required = FEATURES + ["URL", "target"]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required].copy()

for column in FEATURES + ["target"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + ["target", "URL"])

df["target"] = df["target"].astype(int)
df["domain"] = df["URL"].apply(extract_hostname)

df = df.dropna(subset=["domain"])

print("Records:", len(df))
print("Unique hostnames:", df["domain"].nunique())
print("\nTarget distribution:")
print(df["target"].value_counts())

# Split by hostname so a hostname cannot appear in both sets.
X = df[FEATURES]
y = df["target"]
groups = df["domain"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_domains = set(groups.iloc[train_idx])
test_domains = set(groups.iloc[test_idx])

overlap = train_domains.intersection(test_domains)

print("\nTraining records:", len(train_idx))
print("Testing records:", len(test_idx))
print("Training hostnames:", len(train_domains))
print("Testing hostnames:", len(test_domains))
print("Hostname overlap:", len(overlap))

if overlap:
    raise RuntimeError("Domain leakage detected!")

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise RuntimeError(
        "One split contains only one target class. "
        "Try another random_state."
    )

print("\nTraining domain-separated URL model...")

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
roc_auc = roc_auc_score(y_test, probabilities)

print("\n========== DOMAIN-SEPARATED URL EVALUATION ==========")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

results = pd.DataFrame([{
    "evaluation": "hostname_group_split",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "train_records": len(train_idx),
    "test_records": len(test_idx),
    "train_hostnames": len(train_domains),
    "test_hostnames": len(test_domains),
    "hostname_overlap": len(overlap)
}])

results.to_csv(
    "reports/url_domain_validation.csv",
    index=False
)

print("\nSaved: reports/url_domain_validation.csv")
print("\nDomain-separated validation completed.")
