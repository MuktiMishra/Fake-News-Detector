"""
utils.py
Text preprocessing utilities for Fake News Detection.
"""

import re
import string
import pickle
import numpy as np

# Try to import nltk components
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set()

stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove punctuation & digits
    4. Remove extra whitespace
    5. Remove stopwords
    6. Stem words
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords and stem
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in STOP_WORDS and len(w) > 2]

    return " ".join(tokens)


def save_tokenizer(tokenizer, path: str = "model/tokenizer.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {path}")


def load_tokenizer(path: str = "model/tokenizer.pkl"):
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from {path}")
    return tokenizer


def predict_news(text: str, model, tokenizer, max_len: int = 300) -> dict:
    """
    Predicts whether a news article is FAKE or REAL.
    Returns a dict with label and confidence score.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    prob = float(model.predict(padded, verbose=0)[0][0])

    label = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if prob >= 0.5 else 1 - prob

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "raw_probability": round(prob, 4),
    }
