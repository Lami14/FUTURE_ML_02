"""
classifier.py
Trains and evaluates category + priority classifiers for support tickets.
Uses TF-IDF + Logistic Regression (fast, interpretable, production-ready).
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

from preprocessor import TicketPreprocessor


# ── Model config ─────────────────────────────────────────────────────────────

CATEGORY_MODEL_CONFIG = {
    "tfidf": dict(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    ),
    "clf": dict(
        C=1.0,
        max_iter=500,
        random_state=42,
        solver="lbfgs",
    ),
}

PRIORITY_MODEL_CONFIG = {
    "tfidf": dict(
        max_features=3000,
        ngram_range=(1, 3),
        min_df=1,
        sublinear_tf=True,
    ),
    "clf": dict(
        C=0.5,
        max_iter=500,
        random_state=42,
        solver="lbfgs",
    ),
}


class TicketClassifier:
    """
    Two-stage classification system:
      Stage 1 → Category  (billing / technical / account / general_inquiry / shipping)
      Stage 2 → Priority  (high / medium / low)

    Both stages use a TF-IDF → Logistic Regression pipeline.
    """

    PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

    def __init__(self):
        self.preprocessor = TicketPreprocessor()
        self.category_pipeline: Pipeline | None = None
        self.priority_pipeline: Pipeline | None = None
        self.category_encoder = LabelEncoder()
        self.priority_encoder = LabelEncoder()
        self._is_trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, tickets: list[dict]) -> dict:
        """Train both models. Returns evaluation metrics dict."""
        texts = [self.preprocessor.process(t["text"])["token_string"] for t in tickets]
        categories = [t["category"] for t in tickets]
        priorities = [t["priority"] for t in tickets]

        # Train/test split
        (
            X_train, X_test,
            y_cat_train, y_cat_test,
            y_pri_train, y_pri_test,
        ) = train_test_split(
            texts, categories, priorities,
            test_size=0.2, random_state=42, stratify=categories
        )

        print(f"Train: {len(X_train)} | Test: {len(X_test)}")

        # ── Category model ────────────────────────────────────────────────────
        print("\n[1/2] Training category classifier…")
        self.category_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**CATEGORY_MODEL_CONFIG["tfidf"])),
            ("clf", LogisticRegression(**CATEGORY_MODEL_CONFIG["clf"])),
        ])
        self.category_pipeline.fit(X_train, y_cat_train)

        cat_preds = self.category_pipeline.predict(X_test)
        cat_metrics = self._evaluate(y_cat_test, cat_preds, "Category")

        # Cross-val
        cat_cv = cross_val_score(
            self.category_pipeline, texts, categories, cv=5, scoring="f1_weighted"
        )
        cat_metrics["cv_f1_mean"] = float(cat_cv.mean())
        cat_metrics["cv_f1_std"] = float(cat_cv.std())

        # ── Priority model ────────────────────────────────────────────────────
        print("\n[2/2] Training priority classifier…")
        self.priority_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**PRIORITY_MODEL_CONFIG["tfidf"])),
            ("clf", LogisticRegression(**PRIORITY_MODEL_CONFIG["clf"])),
        ])
        self.priority_pipeline.fit(X_train, y_pri_train)

        pri_preds = self.priority_pipeline.predict(X_test)
        pri_metrics = self._evaluate(y_pri_test, pri_preds, "Priority")

        pri_cv = cross_val_score(
            self.priority_pipeline, texts, priorities, cv=5, scoring="f1_weighted"
        )
        pri_metrics["cv_f1_mean"] = float(pri_cv.mean())
        pri_metrics["cv_f1_std"] = float(pri_cv.std())

        self._is_trained = True

        return {
            "category": cat_metrics,
            "priority": pri_metrics,
            "test_size": len(X_test),
            "train_size": len(X_train),
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """Classify a single ticket. Returns full prediction with confidence."""
        if not self._is_trained:
            raise RuntimeError("Call .fit() before .predict()")

        processed = self.preprocessor.process(text)
        token_str = processed["token_string"]
        features = processed["features"]

        # Category prediction + probabilities
        cat_probs = self.category_pipeline.predict_proba([token_str])[0]
        cat_classes = self.category_pipeline.classes_
        cat_label = cat_classes[np.argmax(cat_probs)]
        cat_confidence = float(np.max(cat_probs))

        # Priority prediction + probabilities
        pri_probs = self.priority_pipeline.predict_proba([token_str])[0]
        pri_classes = self.priority_pipeline.classes_
        pri_label = pri_classes[np.argmax(pri_probs)]
        pri_confidence = float(np.max(pri_probs))

        # Override with rule-based boost for clear urgency signals
        pri_label = self._apply_priority_rules(text, pri_label, features)

        return {
            "text": text,
            "category": cat_label,
            "category_confidence": round(cat_confidence, 3),
            "category_probabilities": dict(zip(cat_classes, cat_probs.round(3).tolist())),
            "priority": pri_label,
            "priority_confidence": round(pri_confidence, 3),
            "priority_probabilities": dict(zip(pri_classes, pri_probs.round(3).tolist())),
            "urgency_score": features["urgency_high_score"],
            "features": features,
        }

    def predict_batch(self, tickets: list[dict]) -> list[dict]:
        """Classify multiple tickets efficiently."""
        return [self.predict(t["text"]) for t in tickets]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "models/classifier.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str = "models/classifier.pkl") -> "TicketClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Top features ──────────────────────────────────────────────────────────

    def top_features(self, model: str = "category", n: int = 10) -> dict:
        """Returns top TF-IDF features per class for interpretability."""
        pipeline = self.category_pipeline if model == "category" else self.priority_pipeline
        vectorizer = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        feature_names = vectorizer.get_feature_names_out()

        top = {}
        for i, label in enumerate(clf.classes_):
            coefs = clf.coef_[i]
            idx = np.argsort(coefs)[::-1][:n]
            top[label] = [(feature_names[j], round(float(coefs[j]), 3)) for j in idx]
        return top

    # ── Private helpers ───────────────────────────────────────────────────────

    def _evaluate(self, y_true, y_pred, label: str) -> dict:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        classes = sorted(set(y_true))

        print(f"\n{label} Results:")
        print(f"  Accuracy : {acc:.3f}")
        print(f"  F1 (wtd) : {f1:.3f}")
        print(classification_report(y_true, y_pred))

        return {
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1, 4),
            "report": report,
            "confusion_matrix": cm.tolist(),
            "classes": classes,
        }

    def _apply_priority_rules(self, text: str, predicted: str, features: dict) -> str:
        """Rule-based overrides for strong urgency signals."""
        text_lower = text.lower()

        # Hard escalate to HIGH
        hard_high = [
            "production down", "data loss", "security breach", "hacked",
            "fraud", "legal deadline", "gdpr", "revenue loss", "critical error",
            "completely down", "all users affected",
        ]
        if any(phrase in text_lower for phrase in hard_high):
            return "high"

        # Soft escalate: many urgency signals → bump up one level
        if features["urgency_high_score"] >= 2 and predicted == "low":
            return "medium"
        if features["urgency_high_score"] >= 3 and predicted == "medium":
            return "high"

        # Exclamation marks are a weak urgency signal
        if features["exclamation_count"] >= 2 and predicted == "low":
            return "medium"

        return predicted


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_generator import generate_dataset

    print("=" * 60)
    print("  Support Ticket Classifier — Training Run")
    print("=" * 60)

    # Generate & train
    tickets = generate_dataset(1200)
    clf = TicketClassifier()
    metrics = clf.fit(tickets)
    clf.save("models/classifier.pkl")

    # Show top features
    print("\nTop Category Features:")
    for cat, feats in clf.top_features("category", 5).items():
        words = [f[0] for f in feats]
        print(f"  {cat:20s}: {words}")

    # Demo predictions
    print("\n" + "=" * 60)
    print("  Live Predictions")
    print("=" * 60)
    test_cases = [
        "Our entire production database is DOWN. All 300 users affected. URGENT!",
        "Can I get a receipt for last month's subscription payment?",
        "The export to PDF feature seems broken when I have more than 5 pages.",
        "I was charged $299 twice this month — this is fraud, fix it NOW.",
        "Hi, just wondering if you have an affiliate program?",
    ]
    for text in test_cases:
        result = clf.predict(text)
        print(f"\n📩 {text[:60]}...")
        print(f"   Category : {result['category']:20s} ({result['category_confidence']:.0%})")
        print(f"   Priority : {result['priority']:10s} ({result['priority_confidence']:.0%})")
        print(f"   Urgency  : {'🔴' * result['urgency_score'] or '⚪'}")

    # Save metrics
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print("\nMetrics saved → outputs/metrics.json")
