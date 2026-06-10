import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs using regex
    text = re.sub(r"http\S+", "", text)
    
    # Create translation table for punctuation removal
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    
    # Strip extra whitespace (leading and trailing, then collapse internal spaces)
    text = " ".join(text.split())
    text = text.strip()
    
    return text


def preprocess_series(series: pd.Series) -> pd.Series:
    """Apply clean_text to every element in the series."""
    cleaned_series = series.apply(clean_text)
    return cleaned_series


def build_tfidf_vectorizer():
    """Build and return a TfidfVectorizer with specified parameters."""
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3,
        strip_accents="unicode",
        analyzer="word",
        sublinear_tf=True
    )
    return vectorizer


def fit_transform_tfidf(X_train: pd.Series, X_test: pd.Series) -> tuple:
    """Fit and transform train/test data with TF-IDF."""
    # Preprocess both series
    X_train_cleaned = preprocess_series(X_train)
    X_test_cleaned = preprocess_series(X_test)
    
    # Build vectorizer
    vectorizer = build_tfidf_vectorizer()
    
    # Fit on training data and transform it
    X_train_tfidf = vectorizer.fit_transform(X_train_cleaned)
    
    # Transform test data using fitted vectorizer
    X_test_tfidf = vectorizer.transform(X_test_cleaned)
    
    # Print shapes of resulting matrices
    print(f"Training matrix shape: {X_train_tfidf.shape}")
    print(f"Test matrix shape: {X_test_tfidf.shape}")
    
    return (X_train_tfidf, X_test_tfidf, vectorizer)
