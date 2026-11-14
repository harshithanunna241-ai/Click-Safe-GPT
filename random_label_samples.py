
import pandas as pd

# EMAIL DATASET
email = pd.read_csv("data/email/phishing_email.csv")

print("\n========== EMAIL DATASET ==========")
print("Shape:", email.shape)
print("Missing values:")
print(email.isnull().sum())

for label in sorted(email["label"].dropna().unique()):
    print(f"\nEMAIL LABEL: {label}")

    samples = email.loc[
        email["label"] == label, "text_combined"
    ].sample(
        n=min(5, (email["label"] == label).sum()),
        random_state=42
    )

    for sample in samples:
        print(str(sample)[:350])
        print("-" * 40)


# URL DATASET
url = pd.read_csv(
    "data/urls/LegitPhish Dataset/url_features_extracted1.csv"
)

print("\n========== URL DATASET ==========")
print("Shape:", url.shape)
print("Missing values:")
print(url.isnull().sum())

print("\nURL LABEL DISTRIBUTION:")
print(url["ClassLabel"].value_counts(dropna=False))

for label in sorted(url["ClassLabel"].dropna().unique()):
    print(f"\nURL LABEL: {label}")

    samples = url.loc[
        url["ClassLabel"] == label, ["URL", "ClassLabel"]
    ].sample(
        n=min(10, (url["ClassLabel"] == label).sum()),
        random_state=42
    )

    print(samples.to_string(index=False))
