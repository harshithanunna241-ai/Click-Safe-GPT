import pandas as pd

# SMS dataset
sms = pd.read_csv(
    "data/sms/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

print("\n========== SMS LABELS ==========")
print(sms["label"].value_counts())
print(sms.head())

# Email dataset
email = pd.read_csv("data/email/phishing_email.csv", nrows=5)

print("\n========== EMAIL STRUCTURE ==========")
print("Columns:", email.columns.tolist())
print(email.head())

# URL dataset
url = pd.read_csv(
    "data/urls/LegitPhish Dataset/url_features_extracted1.csv"
)

print("\n========== URL LABELS ==========")
print(url["ClassLabel"].value_counts())
print(url[["URL", "ClassLabel"]].head())