import os
import pandas as pd

datasets = {
    "SMS": "data/sms/SMSSpamCollection",
    "Email": "data/email/phishing_email.csv",
    "URL": "data/urls/LegitPhish Dataset/url_features_extracted1.csv"
}

for name, path in datasets.items():
    print("\n" + "=" * 60)
    print(f"{name} DATASET")
    print("=" * 60)

    if not os.path.exists(path):
        print("File not found:", path)
        continue

    if name == "SMS":
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="latin-1"
        )
    else:
        df = pd.read_csv(path, nrows=5)

    print("\nShape/sample rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDataset preview completed.")