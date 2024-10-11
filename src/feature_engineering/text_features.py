import pandas as pd
import numpy as np
from typing import List, Dict
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузка необходимых ресурсов NLTK
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Инициализация стоп-слов и анализатора настроений
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()


def preprocess_text(text: str) -> List[str]:
    """
    Предобработка текста: токенизация и удаление стоп-слов
    """
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum() and token not in stop_words]


def extract_keywords(texts: List[str], top_n: int = 10) -> List[str]:
    """
    Извлечение ключевых слов из списка текстов
    """
    all_words = [word for text in texts for word in preprocess_text(text)]
    word_freq = Counter(all_words)
    return [word for word, _ in word_freq.most_common(top_n)]


def sentiment_analysis(text: str) -> Dict[str, float]:
    """
    Анализ настроений текста
    """
    return sia.polarity_scores(text)


def tfidf_features(texts: List[str], max_features: int = 100) -> pd.DataFrame:
    """
    Извлечение TF-IDF признаков из текстов
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)


def extract_text_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Извлечение всех текстовых признаков из DataFrame
    """
    texts = df[text_column].tolist()

    # Извлечение ключевых слов
    keywords = extract_keywords(texts)

    # Анализ настроений
    sentiment_scores = df[text_column].apply(sentiment_analysis)
    df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
    df['sentiment_positive'] = sentiment_scores.apply(lambda x: x['pos'])
    df['sentiment_negative'] = sentiment_scores.apply(lambda x: x['neg'])
    df['sentiment_neutral'] = sentiment_scores.apply(lambda x: x['neu'])

    # TF-IDF признаки
    tfidf_features_df = tfidf_features(texts)

    # Объединение всех признаков
    result_df = pd.concat([df, tfidf_features_df], axis=1)

    return result_df, keywords


def add_text_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Добавление текстовых признаков к исходному DataFrame
    """
    df_with_features, keywords = extract_text_features(df, text_column)
    print(f"Extracted keywords: {', '.join(keywords)}")
    return df_with_features