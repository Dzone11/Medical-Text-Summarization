# modules/preprocessing.py

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def handle_missing_values(df, column='transcription'):
    return df.dropna(subset=[column])

def clean_tokenize_stem(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))  # Retain alphanumeric characters
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stopwords
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]  # Apply stemming
    return tokens

def preprocess_text_data(df, column='transcription'):
    df = handle_missing_values(df, column)
    df['clean_tokens'] = df[column].apply(clean_tokenize_stem)
    return df
