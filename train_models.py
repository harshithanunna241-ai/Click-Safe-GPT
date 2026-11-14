
import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.impute import SimpleImputer

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    print(f"\n========== {name} EVALUATION ==========")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))
    print("F1 Score:", round(f1_score(y_test, predictions, zero_division=0), 4))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(
            y_test, model.predict_proba(X_test)[:, 1]
        )
    }


results = []


# --------------------------------------------------
# 1. SMS SPAM DETECTION
# TF-IDF + LOGISTIC REGRESSION
# --------------------------------------------------

print("\nTraining SMS Spam Detection Model...")

sms_train = pd.read_csv("data/splits/sms_train.csv")
sms_test = pd.read_csv("data/splits/sms_test.csv")

sms_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

sms_model.fit(sms_train["text"], sms_train["target"])

joblib.dump(sms_model, "models/sms_model.joblib")

results.append(
    evaluate_model(
        "SMS SPAM DETECTION",
        sms_model,
        sms_test["text"],
        sms_test["target"]
    )
)


# --------------------------------------------------
# 2. EMAIL SUSPICIOUS MESSAGE DETECTION
# TF-IDF + RANDOM FOREST
# --------------------------------------------------

print("\nTraining Email Suspicious Message Model...")

email_train = pd.read_csv(
    "data/splits/email_train.csv",
    low_memory=False
)
email_test = pd.read_csv(
    "data/splits/email_test.csv",
    low_memory=False
)

email_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=40,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

email_model.fit(email_train["text"], email_train["target"])

joblib.dump(email_model, "models/email_model.joblib")

results.append(
    evaluate_model(
        "EMAIL SUSPICIOUS MESSAGE DETECTION",
        email_model,
        email_test["text"],
        email_test["target"]
    )
)


# --------------------------------------------------
# 3. URL PHISHING DETECTION
# RANDOM FOREST + ENGINEERED FEATURES
# --------------------------------------------------

print("\nTraining URL Phishing Detection Model...")

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

url_train = pd.read_csv("data/splits/url_train.csv")
url_test = pd.read_csv("data/splits/url_test.csv")

url_model = Pipeline([
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

url_model.fit(
    url_train[url_features],
    url_train["target"]
)

joblib.dump(
    {
        "model": url_model,
        "features": url_features
    },
    "models/url_model.joblib"
)

results.append(
    evaluate_model(
        "URL PHISHING DETECTION",
        url_model,
        url_test[url_features],
        url_test["target"]
    )
)


# --------------------------------------------------
# SAVE EVALUATION RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "reports/model_evaluation.csv",
    index=False
)

print("\n" + "=" * 60)
print("ALL THREE MODELS TRAINED SUCCESSFULLY")
print("=" * 60)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

print("\nSaved models:")

for filename in os.listdir("models"):
    path = os.path.join("models", filename)
    print(filename, "-", round(os.path.getsize(path) / (1024 * 1024), 2), "MB")

print("\nEvaluation report saved to reports/model_evaluation.csv")
