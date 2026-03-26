import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import clean_text

# =========================
# Load Dataset
# =========================
df = pd.read_csv("data/tickets.csv")

# Rename columns if needed
# df.columns = ['ticket_text', 'category', 'priority']

# =========================
# Preprocessing
# =========================
df['clean_text'] = df['ticket_text'].apply(clean_text)

# =========================
# Feature Extraction
# =========================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])

# =========================
# Encode Labels
# =========================
le_cat = LabelEncoder()
y_cat = le_cat.fit_transform(df['category'])

le_pri = LabelEncoder()
y_pri = le_pri.fit_transform(df['priority'])

# =========================
# Train Category Model
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

model_cat = LogisticRegression(max_iter=200)
model_cat.fit(X_train, y_train)

y_pred = model_cat.predict(X_test)

print("=== Category Model ===")
print(classification_report(y_test, y_pred))

# =========================
# Train Priority Model
# =========================
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_pri, test_size=0.2, random_state=42)

model_pri = LogisticRegression(max_iter=200)
model_pri.fit(X_train_p, y_train_p)

y_pred_p = model_pri.predict(X_test_p)

print("=== Priority Model ===")
print(classification_report(y_test_p, y_pred_p))

# =========================
# Save Models
# =========================
pickle.dump(model_cat, open("model_category.pkl", "wb"))
pickle.dump(model_pri, open("model_priority.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(le_cat, open("label_category.pkl", "wb"))
pickle.dump(le_pri, open("label_priority.pkl", "wb"))

print("✅ Models saved successfully!")
