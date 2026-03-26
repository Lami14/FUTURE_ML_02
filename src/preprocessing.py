import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (runs once)
nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    Clean and preprocess text data.

    خطوات:
    - Lowercase
    - Remove punctuation
    - Remove extra spaces
    - Remove stopwords
    """
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)        # remove punctuation
    text = re.sub(r'\s+', ' ', text)       # remove extra spaces
    
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    
    return " ".join(words)
