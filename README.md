# 🎫 Support Ticket Classifier

**NLP-powered ticket classification system** — automatically categorises support tickets and assigns priority levels to help support teams respond faster and smarter.

---

## ✨ Features

| Feature | Details |
|--------|---------|
| **Text Preprocessing** | Contraction expansion, URL/email/amount normalisation, stop word removal, simple stemming |
| **Category Classification** | 5 classes: billing, technical, account, general_inquiry, shipping |
| **Priority Tagging** | 3 levels: high / medium / low |
| **Confidence Scores** | Per-class probability output for all predictions |
| **Rule-based Override** | Urgency signal boost (e.g. "production down" → always HIGH) |
| **Model Evaluation** | Accuracy, weighted F1, confusion matrix, 5-fold cross-validation |
| **Feature Interpretability** | Top TF-IDF features per class |
| **Interactive Dashboard** | Live HTML demo — classify tickets, view analytics, NLP pipeline |

---

## 📁 Project Structure

```
ticket-classifier/
├── src/
│   ├── data_generator.py    # Synthetic ticket dataset (1200 tickets)
│   ├── preprocessor.py      # NLP cleaning & tokenization pipeline
│   └── classifier.py        # ML models: TF-IDF + Logistic Regression
├── notebooks/
│   └── ticket_classifier.ipynb   # Full Jupyter walkthrough
├── data/
│   └── tickets.json         # Generated training data
├── models/
│   └── classifier.pkl       # Serialised trained model
└── outputs/
    ├── dashboard.html        # Interactive demo
    └── metrics.json          # Model evaluation results
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install scikit-learn numpy

# 2. Generate training data
python src/data_generator.py

# 3. Train the model
python src/classifier.py

# 4. Open dashboard
open outputs/dashboard.html
```

---

## 🧠 ML Architecture

```
Raw Ticket Text
      │
      ▼
┌─────────────────────────────────┐
│       TicketPreprocessor        │
│  • Lowercase + contraction fix  │
│  • Remove HTML / URLs / emails  │
│  • Normalise amounts & numbers  │
│  • Tokenise + stop word removal │
│  • Simple suffix stemming       │
│  • Extract urgency features     │
└───────────────┬─────────────────┘
                │
      ┌─────────┴──────────┐
      ▼                    ▼
┌──────────┐         ┌──────────┐
│ TF-IDF   │         │ TF-IDF   │
│ (5000 f) │         │ (3000 f) │
│ 1-2gram  │         │ 1-3gram  │
└────┬─────┘         └────┬─────┘
     ▼                    ▼
┌──────────┐         ┌──────────┐
│ LogReg   │         │ LogReg   │
│ C=1.0    │         │ C=0.5    │
└────┬─────┘         └────┬─────┘
     ▼                    ▼
  Category             Priority
(5 classes)          (high/med/low)
                          │
                          ▼
                  Rule-based override
                  (urgency signals)
```

---

## 📊 Model Performance

| Model | Accuracy | F1 (weighted) | CV F1 |
|-------|----------|---------------|-------|
| Category Classifier | 100% | 100% | 100% ± 0.000 |
| Priority Classifier | 100% | 100% | 100% ± 0.000 |

> Training on 1200 synthetic tickets with 80/20 train-test split.

---

## 💡 How to Use

### Programmatic API

```python
from src.classifier import TicketClassifier

# Load pre-trained model
clf = TicketClassifier.load('models/classifier.pkl')

# Classify a ticket
result = clf.predict("Our production DB is DOWN. All users affected!")

print(result['category'])           # → 'technical'
print(result['priority'])           # → 'high'
print(result['category_confidence'])  # → 0.89
print(result['urgency_score'])      # → 2
```

### Batch Classification

```python
tickets = [
    {'text': 'I was charged twice this month!'},
    {'text': 'Just wondering about your pricing...'},
]

results = clf.predict_batch(tickets)
```

---

## 🔧 Extending the System

**Add new categories**: Add templates to `TICKET_TEMPLATES` in `data_generator.py` and retrain.

**Add new urgency signals**: Extend the `URGENCY_SIGNALS` dict in `preprocessor.py`.

**Swap the classifier**: Replace `LogisticRegression` in `classifier.py` with `RandomForestClassifier`, `SVC`, or any sklearn estimator.

**Connect to a real database**: Replace `generate_dataset()` with a SQL/API call returning `[{'text': ..., 'category': ..., 'priority': ...}]`.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Scikit-learn** — TF-IDF vectorisation, Logistic Regression, cross-validation
- **NumPy** — numerical operations
- **Jupyter Notebook** — interactive exploration
- **HTML/JS** — interactive dashboard (zero dependencies)

---

## 📝 Skills Demonstrated

- ✅ Text preprocessing & NLP pipeline design
- ✅ Multi-class classification with scikit-learn
- ✅ Feature engineering (TF-IDF, urgency signals, meta-features)
- ✅ Model evaluation (accuracy, F1, confusion matrix, cross-validation)
- ✅ Model serialisation and loading
- ✅ Rule-based + ML hybrid systems
- ✅ End-to-end ML project structure
- ✅ Interactive demo / data visualisation
