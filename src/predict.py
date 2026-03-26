import pickle
from preprocessing import clean_text

# Load models
model_cat = pickle.load(open("model_category.pkl", "rb"))
model_pri = pickle.load(open("model_priority.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le_cat = pickle.load(open("label_category.pkl", "rb"))
le_pri = pickle.load(open("label_priority.pkl", "rb"))

# =========================
# Rule-Based Priority Boost
# =========================
def boost_priority(text, predicted_priority):
    urgent_keywords = ['urgent', 'asap', 'immediately', 'failed', 'error', 'critical']

    if any(word in text.lower() for word in urgent_keywords):
        return "High"
    
    return predicted_priority

# =========================
# Prediction Function
# =========================
def predict_ticket(text: str):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    # Predictions
    cat_pred = model_cat.predict(vec)[0]
    pri_pred = model_pri.predict(vec)[0]

    # Convert labels
    category = le_cat.inverse_transform([cat_pred])[0]
    priority = le_pri.inverse_transform([pri_pred])[0]

    # Boost priority
    priority = boost_priority(text, priority)

    return {
        "text": text,
        "category": category,
        "priority": priority
    }

# =========================
# Test
# =========================
if __name__ == "__main__":
    sample = "System error and payment failed urgently"
    result = predict_ticket(sample)

    print("\nPrediction Result:")
    print(result)
