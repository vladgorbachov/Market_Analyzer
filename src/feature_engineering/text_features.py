# src/feature_engineering/text_features.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка необходимых ресурсов NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logger.info("Downloading necessary NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

# Инициализация стоп-слов и анализатора настроений
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

def preprocess_text(text: str) -> List[str]:
    """
    Предобработка текста: токенизация и удаление стоп-слов.

    :param text: Исходный текст.
    :return: Список токенов после предобработки.
    """
    try:
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token.isalnum() and token not in stop_words]
    except Exception as e:
        logger.error(f"Error in text preprocessing: {str(e)}")
        return []

def extract_keywords(texts: List[str], top_n: int = 10) -> List[str]:
    """
    Извлечение ключевых слов из списка текстов.

    :param texts: Список текстов.
    :param top_n: Количество топ ключевых слов для извлечения.
    :return: Список топ ключевых слов.
    """
    logger.info(f"Extracting top {top_n} keywords")
    try:
        all_words = [word for text in texts for word in preprocess_text(text)]
        word_freq = Counter(all_words)
        return [word for word, _ in word_freq.most_common(top_n)]
    except Exception as e:
        logger.error(f"Error in keyword extraction: {str(e)}")
        return []

def sentiment_analysis(text: str) -> Dict[str, float]:
    """
    Анализ настроений текста.

    :param text: Исходный текст.
    :return: Словарь с оценками настроений.
    """
    try:
        return sia.polarity_scores(text)
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}

def tfidf_features(texts: List[str], max_features: int = 100) -> pd.DataFrame:
    """
    Извлечение TF-IDF признаков из текстов.

    :param texts: Список текстов.
    :param max_features: Максимальное количество признаков для извлечения.
    :return: DataFrame с TF-IDF признаками.
    """
    logger.info(f"Extracting TF-IDF features with max_features={max_features}")
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    except Exception as e:
        logger.error(f"Error in TF-IDF feature extraction: {str(e)}")
        return pd.DataFrame()

def extract_text_features(df: pd.DataFrame, text_column: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Извлечение всех текстовых признаков из DataFrame.

    :param df: Исходный DataFrame.
    :param text_column: Название колонки с текстовыми данными.
    :return: Кортеж из DataFrame с извлеченными признаками и списка ключевых слов.
    """
    logger.info(f"Extracting text features from column: {text_column}")
    try:
        texts = df[text_column].fillna('').tolist()

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
    except Exception as e:
        logger.error(f"Error in text feature extraction: {str(e)}")
        return df, []

def add_text_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Добавление текстовых признаков к исходному DataFrame.

    :param df: Исходный DataFrame.
    :param text_column: Название колонки с текстовыми данными.
    :return: DataFrame с добавленными текстовыми признаками.
    """
    logger.info(f"Adding text features to DataFrame")
    try:
        df_with_features, keywords = extract_text_features(df, text_column)
        logger.info(f"Extracted keywords: {', '.join(keywords)}")
        return df_with_features
    except Exception as e:
        logger.error(f"Error adding text features: {str(e)}")
        return df

def calculate_text_length(text: str) -> int:
    """
    Рассчитывает длину текста.

    :param text: Исходный текст.
    :return: Длина текста.
    """
    return len(text)

def calculate_word_count(text: str) -> int:
    """
    Рассчитывает количество слов в тексте.

    :param text: Исходный текст.
    :return: Количество слов.
    """
    return len(text.split())

def add_basic_text_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Добавляет базовые текстовые признаки к DataFrame.

    :param df: Исходный DataFrame.
    :param text_column: Название колонки с текстовыми данными.
    :return: DataFrame с добавленными базовыми текстовыми признаками.
    """
    logger.info(f"Adding basic text features to DataFrame")
    try:
        df = df.copy()
        df['text_length'] = df[text_column].apply(calculate_text_length)
        df['word_count'] = df[text_column].apply(calculate_word_count)
        return df
    except Exception as e:
        logger.error(f"Error adding basic text features: {str(e)}")
        return df

# Добавьте дополнительные функции для обработки текста по мере необходимости