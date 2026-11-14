
from backend.app.services.risk_engine import analyze_prediction


def test_safe_result():
    result = analyze_prediction("SMS", 0.05)

    assert result["risk_level"] == "Safe"
    assert result["risk_score"] == 5


def test_suspicious_result():
    result = analyze_prediction("EMAIL", 0.50)

    assert result["risk_level"] == "Suspicious"
    assert result["risk_score"] == 50


def test_dangerous_result():
    result = analyze_prediction("URL", 0.90)

    assert result["risk_level"] == "Dangerous"
    assert result["risk_score"] == 90


def test_indicator_adjustment():
    result = analyze_prediction(
        "SMS",
        0.50,
        indicators=[
            "Contains suspicious payment request",
            "Requests an OTP"
        ]
    )

    assert result["risk_score"] == 60
    assert result["indicator_count"] == 2


def test_recommendations():
    result = analyze_prediction("URL", 0.95)

    assert len(result["recommendations"]) > 0
    assert result["category"] == "URL"


def test_invalid_probability():
    try:
        analyze_prediction("SMS", 1.5)
    except ValueError:
        return

    raise AssertionError("Invalid probability was accepted")


if __name__ == "__main__":
    print("Running Risk Engine Tests...")

    tests = [
        test_safe_result,
        test_suspicious_result,
        test_dangerous_result,
        test_indicator_adjustment,
        test_recommendations,
        test_invalid_probability
    ]

    for test in tests:
        test()
        print(test.__name__, "PASSED")

    print("\nAll Risk Engine Tests Passed!")
