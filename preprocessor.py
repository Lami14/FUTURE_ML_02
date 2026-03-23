"""
preprocessor.py
Text cleaning and tokenization pipeline for support tickets.
Uses regex + basic NLP (no heavy downloads required for demo).
"""

import re
import string
from typing import Optional

# ── Stop words (curated for support ticket domain) ──────────────────────────

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "i", "me", "my", "we", "our", "you", "your", "they", "their", "it",
    "its", "this", "that", "these", "those", "he", "she", "him", "her",
    "there", "here", "when", "where", "how", "what", "who", "which",
    "just", "very", "so", "too", "also", "about", "up", "out", "if",
    "then", "than", "more", "some", "any", "all", "no", "not", "hi",
    "hello", "dear", "please", "thank", "thanks", "sincerely", "regards",
    "good", "day", "whom", "concern", "looking", "forward", "response",
}

# ── Urgency signal words (boost priority detection) ──────────────────────────

URGENCY_SIGNALS = {
    "high": {
        "urgent", "immediately", "asap", "critical", "emergency", "now",
        "right away", "halted", "breach", "loss", "lost", "fraud",
        "unauthorized", "deadline", "today", "hours", "suspended",
        "crashing", "down", "missing", "hacked", "locked out",
    },
    "low": {
        "curious", "wondering", "whenever", "sometime", "eventually",
        "nice to have", "suggestion", "would be great", "minor",
    },
}


class TicketPreprocessor:
    """
    Cleans and tokenizes raw support ticket text.

    Pipeline:
        1. Lowercase
        2. Expand common contractions
        3. Remove HTML / special chars
        4. Normalize whitespace
        5. Tokenize
        6. Remove stop words
        7. Simple stemming (suffix stripping)
    """

    CONTRACTIONS = {
        "can't": "cannot", "won't": "will not", "don't": "do not",
        "didn't": "did not", "doesn't": "does not", "isn't": "is not",
        "wasn't": "was not", "weren't": "were not", "haven't": "have not",
        "hasn't": "has not", "hadn't": "had not", "couldn't": "could not",
        "wouldn't": "would not", "shouldn't": "should not", "i'm": "i am",
        "i've": "i have", "i'll": "i will", "i'd": "i would",
        "it's": "it is", "that's": "that is", "there's": "there is",
        "they're": "they are", "we're": "we are", "you're": "you are",
        "he's": "he is", "she's": "she is", "let's": "let us",
    }

    COMMON_SUFFIXES = ("ing", "tion", "tions", "ed", "er", "ers", "ly", "ness")

    def clean(self, text: str) -> str:
        """Full cleaning pipeline, returns cleaned string."""
        text = text.lower()
        text = self._expand_contractions(text)
        text = re.sub(r"<[^>]+>", " ", text)           # HTML tags
        text = re.sub(r"http\S+|www\.\S+", " URL ", text)  # URLs
        text = re.sub(r"\S+@\S+", " EMAIL ", text)      # emails
        text = re.sub(r"\$[\d,]+\.?\d*", " AMOUNT ", text)  # dollar amounts
        text = re.sub(r"\b\d{3,}\b", " NUMBER ", text)   # long numbers
        text = re.sub(r"[^\w\s]", " ", text)             # punctuation
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str, remove_stops: bool = True,
                 stem: bool = True) -> list[str]:
        """Tokenize cleaned text into word list."""
        tokens = text.split()
        if remove_stops:
            tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
        if stem:
            tokens = [self._simple_stem(t) for t in tokens]
        return tokens

    def extract_features(self, text: str) -> dict:
        """Extract meta-features useful for priority classification."""
        text_lower = text.lower()
        return {
            "char_count": len(text),
            "word_count": len(text.split()),
            "exclamation_count": text.count("!"),
            "question_count": text.count("?"),
            "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "urgency_high_score": sum(
                1 for word in URGENCY_SIGNALS["high"]
                if word in text_lower
            ),
            "urgency_low_score": sum(
                1 for word in URGENCY_SIGNALS["low"]
                if word in text_lower
            ),
        }

    def process(self, text: str) -> dict:
        """Full pipeline: returns cleaned text, tokens, and features."""
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)
        features = self.extract_features(text)
        return {
            "original": text,
            "cleaned": cleaned,
            "tokens": tokens,
            "token_string": " ".join(tokens),  # for TF-IDF vectorizer
            "features": features,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _expand_contractions(self, text: str) -> str:
        for contraction, expansion in self.CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        return text

    def _simple_stem(self, word: str) -> str:
        """Lightweight suffix-stripping (no NLTK download required)."""
        if len(word) <= 4:
            return word
        for suffix in self.COMMON_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                return word[: -len(suffix)]
        return word


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preprocessor = TicketPreprocessor()
    samples = [
        "I was charged twice! Please refund immediately — this is urgent!",
        "Hi, can't access my account. It's been 3 days & I'm very frustrated.",
        "Just wondering when my subscription renews.",
    ]
    for s in samples:
        result = preprocessor.process(s)
        print(f"\nOriginal : {result['original']}")
        print(f"Cleaned  : {result['cleaned']}")
        print(f"Tokens   : {result['tokens'][:8]}...")
        print(f"Features : {result['features']}")
