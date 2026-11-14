
from backend.app.services.url_feature_extractor import (
    extract_url_features
)

test_urls = [
    "https://www.google.com",
    "http://192.168.1.10/download.exe",
    "https://example.com/login?user=test"
]

for url in test_urls:
    print("\nURL:", url)

    features = extract_url_features(url)

    for name, value in features.items():
        print(f"{name}: {value}")

    assert len(features) == 16
    assert all(isinstance(value, (int, float))
               for value in features.values())

print("\nAll URL feature extraction tests passed!")
