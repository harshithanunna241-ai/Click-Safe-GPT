
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "data/splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def split_and_save(name, df, feature_columns):
    print(f"\n========== {name} DATASET SPLITTING ==========")

    # Keep only the required columns
    data = df[feature_columns + ["target"]].copy()

    # Ensure target is integer
    data["target"] = pd.to_numeric(data["target"], errors="coerce")
    data = data.dropna(subset=["target"])
    data["target"] = data["target"].astype(int)

    # Remove invalid numeric values from URL features
    for column in feature_columns:
        if column != "text":
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    X = data[feature_columns]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    train_data = X_train.copy()
    train_data["target"] = y_train

    test_data = X_test.copy()
    test_data["target"] = y_test

    train_path = f"{OUTPUT_DIR}/{name.lower()}_train.csv"
    test_path = f"{OUTPUT_DIR}/{name.lower()}_test.csv"

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print("Total records:", len(data))
    print("Training records:", len(train_data))
    print("Testing records:", len(test_data))

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting target distribution:")
    print(y_test.value_counts().sort_index())

    print("\nSaved:", train_path)
    print("Saved:", test_path)


# 1. SMS DATASET
sms = pd.read_csv("data/processed/sms_clean.csv")

split_and_save(
    "SMS",
    sms,
    ["text"]
)


# 2. EMAIL DATASET
email = pd.read_csv(
    "data/processed/email_clean.csv",
    low_memory=False
)

split_and_save(
    "EMAIL",
    email,
    ["text"]
)


# 3. URL DATASET
url = pd.read_csv("data/processed/url_clean.csv")

url_features = [
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

split_and_save(
    "URL",
    url,
    url_features
)


print("\n" + "=" * 55)
print("ALL DATASETS SPLIT SUCCESSFULLY")
print("=" * 55)

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    path = os.path.join(OUTPUT_DIR, filename)
    print(
        filename,
        "-",
        round(os.path.getsize(path) / (1024 * 1024), 2),
        "MB"
    )
