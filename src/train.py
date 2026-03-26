import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import clean_text

RANDOM_STATE = 42

# =========================
# Load Data
# =========================
df = pd.read_csv("data/tickets.csv")

# =========================
# Preprocess
# =========================
df['clean_text'] = df['ticket_text'].apply(clean_text)

# =========================
# Feature Engineering
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
# Split Data
# =========================
X_train, X_test, y_cat_train, y_cat_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=RANDOM_STATE
)

# =========================
# Train Category Model
# =========================
model_cat = LogisticRegression(max_iter=300)
model_cat.fit(X_train, y_cat_train)

y_pred_cat = model_cat.predict(X_test)

print("\n=== CATEGORY MODEL ===")
print(classification_report(y_cat_test, y_pred_cat))

# =========================
# Train Priority Model
# =========================
X_train_p, X_test_p, y_pri_train, y_pri_test = train_test_split(
    X, y_pri, test_size=0.2, random_state=RANDOM_STATE
)

model_pri = LogisticRegression(max_iter=300)
model_pri.fit(X_train_p, y_pri_train)

y_pred_pri = model_pri.predict(X_test_p)

print("\n=== PRIORITY MODEL ===")
print(classification_report(y_pri_test, y_pred_pri))

# =========================
# Save Artifacts
# =========================
pickle.dump(model_cat, open("model_category.pkl", "wb"))
pickle.dump(model_pri, open("model_priority.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(le_cat, open("label_category.pkl", "wb"))
pickle.dump(le_pri, open("label_priority.pkl", "wb"))

print("\n✅ Models saved successfully!")
