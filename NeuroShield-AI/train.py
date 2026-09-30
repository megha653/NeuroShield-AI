from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).parent

DATA = BASE / "data" / "messages.csv"
MODELS = BASE / "models"
ARTIFACTS = BASE / "artifacts"

MODELS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("NEUROSHIELD AI - MODEL TRAINING PIPELINE")
print("=" * 60)

df = pd.read_csv(DATA)

print(f"\n[+] Original dataset size: {len(df)}")


# ============================================================
# DATA CLEANING
# ============================================================

required_columns = {"text", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "Dataset must contain 'text' and 'label' columns."
    )


# Remove missing values
df = df.dropna(subset=["text", "label"])


# Convert text to string and remove extra spaces
df["text"] = df["text"].astype(str).str.strip()


# Remove empty messages
df = df[df["text"] != ""]


# Remove duplicate rows
df = df.drop_duplicates(subset=["text", "label"])


print(f"[+] Cleaned dataset size: {len(df)}")


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nCLASS DISTRIBUTION")
print("-" * 60)

class_distribution = df["label"].value_counts()

print(class_distribution)


# Save class distribution
class_distribution.to_csv(
    ARTIFACTS / "class_distribution.csv",
    header=["count"],
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("\nDATA SPLIT")
print("-" * 60)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# TF-IDF FEATURE EXTRACTION
# ============================================================

print("\n[+] Extracting TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_features=5000,
    sublinear_tf=True,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(
    f"[+] Number of TF-IDF features: "
    f"{len(vectorizer.get_feature_names_out())}"
)


# ============================================================
# DEFINE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=42,
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}


# ============================================================
# TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)


for model_name, model in models.items():

    print(f"\nTraining: {model_name}")
    print("-" * 60)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }
    )

    trained_models[model_name] = model

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False,
).reset_index(drop=True)


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)


# Save comparison
results_df.to_csv(
    ARTIFACTS / "model_comparison.csv",
    index=False,
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]


print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Selected model: {best_model_name}")
print(
    f"F1 Score: "
    f"{results_df.iloc[0]['F1 Score']:.3f}"
)


# ============================================================
# FINAL BEST-MODEL EVALUATION
# ============================================================

best_predictions = best_model.predict(
    X_test_tfidf
)

report = classification_report(
    y_test,
    best_predictions,
    digits=3,
    zero_division=0,
)

cm = confusion_matrix(
    y_test,
    best_predictions,
)


print("\nCLASSIFICATION REPORT")
print("-" * 60)
print(report)

print("CONFUSION MATRIX")
print("-" * 60)
print(cm)


# ============================================================
# SAVE EVALUATION REPORT
# ============================================================

evaluation_text = f"""
NEUROSHIELD AI
MODEL EVALUATION REPORT
============================================================

Dataset Size:
{len(df)}

Training Samples:
{len(X_train)}

Testing Samples:
{len(X_test)}

TF-IDF Features:
{len(vectorizer.get_feature_names_out())}

Best Model:
{best_model_name}

MODEL COMPARISON
------------------------------------------------------------
{results_df.to_string(index=False)}

CLASSIFICATION REPORT
------------------------------------------------------------
{report}

CONFUSION MATRIX
------------------------------------------------------------
{cm}
"""


(ARTIFACTS / "evaluation.txt").write_text(
    evaluation_text,
    encoding="utf-8",
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    MODELS / "text_model.joblib",
)

joblib.dump(
    vectorizer,
    MODELS / "tfidf.joblib",
)


# Save model name separately
(MODELS / "best_model.txt").write_text(
    best_model_name,
    encoding="utf-8",
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"\n[+] Best model saved: "
    f"models/text_model.joblib"
)

print(
    f"[+] TF-IDF vectorizer saved: "
    f"models/tfidf.joblib"
)

print(
    f"[+] Model comparison saved: "
    f"artifacts/model_comparison.csv"
)

print(
    f"[+] Evaluation report saved: "
    f"artifacts/evaluation.txt"
)

print(
    f"\n[+] NeuroShield selected "
    f"{best_model_name} as the best model."
)