import pandas as pd

# EMAIL
email = pd.read_csv("data/email/phishing_email.csv")

print("\n========== EMAIL LABEL DISTRIBUTION ==========")
print(email["label"].value_counts(dropna=False))

print("\n========== EMAIL EXAMPLES BY LABEL ==========")

for label in sorted(email["label"].dropna().unique()):
    print(f"\nLABEL: {label}")
    samples = email.loc[email["label"] == label, "text_combined"].head(3)
    for sample in samples:
        print(str(sample)[:500])
        print("-" * 40)

# URL
url = pd.read_csv(
    "data/urls/LegitPhish Dataset/url_features_extracted1.csv"
)

print("\n========== URL EXAMPLES BY LABEL ==========")

for label in sorted(url["ClassLabel"].dropna().unique()):
    print(f"\nLABEL: {label}")
    samples = url.loc[url["ClassLabel"] == label, "URL"].head(5)
    print(samples.to_string(index=False))