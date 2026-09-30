from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


# PATHS
BASE = Path(__file__).parent
DATA = BASE / "data" / "messages.csv"
MODELS = BASE / "models"
ARTIFACTS = BASE / "artifacts"

MODELS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)


# SETTINGS
MAX_WORDS = 10000
MAX_LENGTH = 80
EMBEDDING_DIM = 64
BATCH_SIZE = 32
EPOCHS = 8


print("=" * 60)
print("NEUROSHIELD AI - DEEP LEARNING NLP")
print("=" * 60)


# LOAD DATA
df = pd.read_csv(DATA)
df = df.dropna(subset=["text", "label"])
df = df.drop_duplicates(subset=["text"])

texts = df["text"].astype(str).values
labels = df["label"].astype(int).values

print(f"\n[+] Dataset size: {len(df)}")


# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


# TRAIN / VALIDATION SPLIT
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)

print(f"[+] Training samples:   {len(X_train)}")
print(f"[+] Validation samples: {len(X_val)}")
print(f"[+] Testing samples:    {len(X_test)}")


# TOKENIZER
tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)


def prepare_sequences(text_data):
    sequences = tokenizer.texts_to_sequences(text_data)

    return pad_sequences(
        sequences,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )


X_train_seq = prepare_sequences(X_train)
X_val_seq = prepare_sequences(X_val)
X_test_seq = prepare_sequences(X_test)


# MODEL
model = Sequential([
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LENGTH
    ),

    Bidirectional(
        LSTM(64)
    ),

    Dropout(0.40),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.30),

    Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("\nMODEL ARCHITECTURE")
print("-" * 60)
model.summary()


# EARLY STOPPING
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


# TRAIN
print("\n[+] Training BiLSTM model...\n")

history = model.fit(
    X_train_seq,
    y_train,
    validation_data=(
        X_val_seq,
        y_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)


# TEST PREDICTIONS
probabilities = model.predict(
    X_test_seq,
    verbose=0
).flatten()

predictions = (
    probabilities >= 0.5
).astype(int)


# METRICS
accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    predictions
)


print("\n" + "=" * 60)
print("DEEP LEARNING RESULTS")
print("=" * 60)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)


# SAVE MODEL
model.save(
    MODELS / "deep_text_model.keras"
)


# SAVE TOKENIZER
tokenizer_json = tokenizer.to_json()

(MODELS / "deep_tokenizer.json").write_text(
    tokenizer_json,
    encoding="utf-8"
)


# SAVE CONFIG
config = {
    "max_words": MAX_WORDS,
    "max_length": MAX_LENGTH,
    "embedding_dim": EMBEDDING_DIM
}

(MODELS / "deep_config.json").write_text(
    json.dumps(config, indent=4),
    encoding="utf-8"
)


# SAVE RESULTS
results = pd.DataFrame([
    {
        "Model": "BiLSTM",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
])

results.to_csv(
    ARTIFACTS / "deep_model_results.csv",
    index=False
)


print("\n[+] Deep model saved.")
print("[+] Tokenizer saved.")
print("[+] Evaluation results saved.")
print("\n[+] NeuroShield Deep Learning training complete.")