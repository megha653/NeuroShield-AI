from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import extract_features


BASE = Path(__file__).parent

DATA = BASE / "data" / "messages.csv"
MODELS = BASE / "models"
ARTIFACTS = BASE / "artifacts"

MODELS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)


FEATURE_NAMES = [
    "message_length",
    "word_count",
    "digit_count",
    "digit_ratio",
    "uppercase_ratio",
    "special_character_count",
    "url_count",
    "urgency_word_count",
]


print("=" * 60)
print("NEUROSHIELD - UNSUPERVISED LEARNING")
print("=" * 60)


df = pd.read_csv(DATA)

df = df.dropna(
    subset=["text", "label"]
)


features = np.array(
    [
        extract_features(text)
        for text in df["text"]
    ]
)


feature_df = pd.DataFrame(
    features,
    columns=FEATURE_NAMES
)

feature_df["label"] = (
    df["label"]
    .astype(int)
    .values
)


feature_df.to_csv(
    ARTIFACTS / "engineered_features.csv",
    index=False
)


# Train anomaly detector using normal messages only
normal_features = features[
    df["label"].astype(int).values == 0
]


scaler = StandardScaler()

normal_scaled = scaler.fit_transform(
    normal_features
)


model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
)

model.fit(
    normal_scaled
)


joblib.dump(
    model,
    MODELS / "anomaly_model.joblib"
)

joblib.dump(
    scaler,
    MODELS / "anomaly_scaler.joblib"
)


print(
    f"Dataset: {len(df)} messages"
)

print(
    f"Engineered features: {len(FEATURE_NAMES)}"
)

print(
    "Isolation Forest trained successfully."
)

print(
    "Unsupervised learning module complete."
)