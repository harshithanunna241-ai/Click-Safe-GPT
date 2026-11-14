
import os
import pandas as pd

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def show_report(name, df):
    print(f"\n========== {name} ==========")
    print("Total records:", len(df))
    print("\nTarget distribution:")
    print(df["target"].value_counts().sort_index())
    print("\nMissing values:")
    print(df.isnull().sum().sum())


# 1. SMS PREPROCESSING
print("\nProcessing SMS dataset...")

sms = pd.read_csv(
    "data/sms/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="latin-1"
)

sms = sms.dropna(subset=["label", "text"])
sms["text"] = sms["text"].astype(str).str.strip()

sms["target"] = sms["label"].map({
    "ham": 0,
    "spam": 1
})

sms = sms.dropna(subset=["target"])
sms["target"] = sms["target"].astype(int)
sms = sms[sms["text"] != ""]
sms = sms.drop_duplicates(subset=["text", "target"])

sms[["text", "target"]].to_csv(
    f"{OUTPUT_DIR}/sms_clean.csv",
    index=False
)

show_report("SMS CLEANED DATASET", sms)


# 2. EMAIL PREPROCESSING
print("\nProcessing Email dataset...")

email = pd.read_csv(
    "data/email/phishing_email.csv",
    low_memory=False
)

required_email_columns = {"text_combined", "label"}

if not required_email_columns.issubset(email.columns):
    raise ValueError("Unexpected Email dataset columns")

email = email[["text_combined", "label"]].copy()
email["text_combined"] = email["text_combined"].fillna("").astype(str).str.strip()
email["label"] = pd.to_numeric(email["label"], errors="coerce")

email = email.dropna(subset=["label"])
email = email[email["label"].isin([0, 1])]
email = email[email["text_combined"] != ""]

email["target"] = email["label"].astype(int)

email = email.rename(columns={
    "text_combined": "text"
})

email = email[["text", "target"]]
email = email.drop_duplicates(subset=["text", "target"])

email.to_csv(
    f"{OUTPUT_DIR}/email_clean.csv",
    index=False
)

show_report("EMAIL CLEANED DATASET", email)


# 3. URL PREPROCESSING
print("\nProcessing URL dataset...")

url = pd.read_csv(
    "data/urls/LegitPhish Dataset/url_features_extracted1.csv"
)

url = url.dropna(subset=["URL", "ClassLabel"]).copy()

url["ClassLabel"] = pd.to_numeric(
    url["ClassLabel"], errors="coerce"
)

url = url.dropna(subset=["ClassLabel"])
url = url[url["ClassLabel"].isin([0, 1])]

url["URL"] = url["URL"].astype(str).str.strip()
url = url[url["URL"] != ""]

# Original dataset:
# 0 = malicious
# 1 = legitimate
# Standard target:
# 1 = malicious
# 0 = legitimate

url["target"] = (1 - url["ClassLabel"]).astype(int)

url = url.drop_duplicates(subset=["URL", "target"])

url.to_csv(
    f"{OUTPUT_DIR}/url_clean.csv",
    index=False
)

show_report("URL CLEANED DATASET", url)


# FINAL SUMMARY
print("\n" + "=" * 55)
print("ALL DATASET PREPROCESSING COMPLETED")
print("=" * 55)

print("\nGenerated files:")

for filename in os.listdir(OUTPUT_DIR):
    path = os.path.join(OUTPUT_DIR, filename)
    print(filename, "-", round(os.path.getsize(path) / (1024 * 1024), 2), "MB")
