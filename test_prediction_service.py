
import pandas as pd

from backend.app.services.prediction_service import (
    PredictionService,
    URL_FEATURES
)


def main():

    print("\nInitializing ClickSafeGPT Prediction Service...")

    service = PredictionService()

    print("\n========== SMS TEST ==========")

    sms_result = service.predict_sms(
        "Congratulations! You have won a free prize. "
        "Click this link now to claim your reward."
    )

    print(sms_result)

    assert sms_result["category"] == "SMS"
    assert 0 <= sms_result["risk_score"] <= 100

    print("\n========== EMAIL TEST ==========")

    email_result = service.predict_email(
        "Dear customer, your account requires verification. "
        "Please review the unexpected request carefully."
    )

    print(email_result)

    assert email_result["category"] == "EMAIL"
    assert 0 <= email_result["risk_score"] <= 100

    print("\n========== URL MODEL INTEGRATION TEST ==========")

    # Use a real held-out row to verify model integration.
    # This does not claim to test raw URL feature extraction.
    url_test = pd.read_csv("data/splits/url_test.csv")

    sample_features = (
        url_test.iloc[0][URL_FEATURES].to_dict()
    )

    url_result = service.predict_url_from_features(
        sample_features
    )

    print(url_result)

    assert url_result["category"] == "URL"
    assert 0 <= url_result["risk_score"] <= 100

    print("\n" + "=" * 55)
    print("ALL THREE MODEL INTEGRATION TESTS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()
