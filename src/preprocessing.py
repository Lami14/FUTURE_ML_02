import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    Clean and normalize text for NLP processing.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]

    return " ".join(tokens)
