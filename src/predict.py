import pickle
from preprocessing import clean_text

# =========================
# Load Models
# =========================
model_cat = pickle.load(open("model_category.pkl", "rb"))
model_pri = pickle.load(open("model_priority.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le_cat = pickle.load(open("label_category.pkl", "rb"))
le_pri = pickle.load(open("label_priority.pkl", "rb"))

# =========================
# Rule-Based Priority Boost
# =========================
def boost_priority(text, predicted_priority):
    urgent_words = ['urgent', 'asap', 'immediately', 'failed', 'error', 'issue']
    
    if any(word in text.lower() for word in urgent_words):
        return "High"
    
    return predicted_priority

# =========================
# Prediction Function
# =========================
def predict_ticket(text: str):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    category = le_cat.inverse_transform(model_cat.predict(vectorized))[0]
    priority = le_pri.inverse_transform(model_pri.predict(vectorized))[0]

    # Apply rule-based boost
    priority = boost_priority(text, priority)

    return category, priority


# =========================
# Example Test
# =========================
if __name__ == "__main__":
    sample = "My payment failed and I need help urgently"
    cat, pri = predict_ticket(sample)
    
    print("Ticket:", sample)
    print("Predicted Category:", cat)
    print("Predicted Priority:", pri)
