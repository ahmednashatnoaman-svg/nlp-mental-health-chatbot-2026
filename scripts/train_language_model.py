"""
Script — Train and save the Language Detection model (Module 1).
Run once: python scripts/train_language_model.py
Saves: models/language_detector.pkl
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SAVE_PATH = ROOT / "models" / "language_detector.pkl"


def main():
    print("Loading papluca/language-identification …")
    ds = load_dataset("papluca/language-identification")

    X_train = ds["train"]["text"]
    y_train = ds["train"]["labels"]
    X_val   = ds["validation"]["text"]
    y_val   = ds["validation"]["labels"]
    X_test  = ds["test"]["text"]
    y_test  = ds["test"]["labels"]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Languages: {sorted(set(y_train))}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=150_000,
            sublinear_tf=True,
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000), cv=3,
        )),
    ])

    print("Training …")
    pipeline.fit(X_train, y_train)

    val_acc = accuracy_score(y_val, pipeline.predict(X_val))
    print(f"Validation accuracy: {val_acc:.4f}")

    y_pred = pipeline.predict(X_test)
    print("\nTest set report:")
    print(classification_report(y_test, y_pred))

    SAVE_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, SAVE_PATH)
    print(f"\nModel saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
