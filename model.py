import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

def train_model():
    df = pd.read_csv('../data/tickets.csv')
    df['clean_text'] = df['text'].apply(clean_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['category']

    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer

if __name__ == "__main__":
    model, vectorizer = train_model()
    print("Model trained successfully!")
