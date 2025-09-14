"""
Sentiment Analysis Module
Provides functions for analyzing sentiment in employee messages
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Try importing sentiment analysis libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    if VADER_AVAILABLE:
        analyzer = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text data
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def analyze_sentiment_simple(text: str) -> str:
    """
    Simple sentiment analysis using keyword matching
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Sentiment label ('Positive', 'Negative', 'Neutral')
    """
    if pd.isna(text) or text == "":
        return 'Neutral'
    
    text = str(text).lower()
    
    # Positive keywords
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
        'happy', 'pleased', 'satisfied', 'perfect', 'awesome', 'brilliant',
        'outstanding', 'exceptional', 'superb', 'magnificent', 'terrific',
        'love', 'like', 'enjoy', 'appreciate', 'thank', 'thanks', 'grateful',
        'success', 'successful', 'win', 'winning', 'best', 'better', 'improve',
        'excited', 'thrilled', 'delighted', 'optimistic', 'positive'
    ]
    
    # Negative keywords
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'frustrated',
        'disappointed', 'sad', 'upset', 'annoyed', 'irritated', 'furious',
        'disgusted', 'outraged', 'appalled', 'concerned', 'worried', 'problem',
        'issue', 'error', 'fail', 'wrong', 'broken', 'difficult', 'hard',
        'worst', 'worse', 'decline', 'decrease', 'loss', 'negative', 'stress',
        'stressed', 'overwhelmed', 'crisis', 'urgent', 'critical'
    ]
    
    positive_score = sum(1 for word in positive_words if word in text)
    negative_score = sum(1 for word in negative_words if word in text)
    
    if positive_score > negative_score:
        return 'Positive'
    elif negative_score > positive_score:
        return 'Negative'
    else:
        return 'Neutral'


def analyze_sentiment_vader(text: str) -> str:
    """
    Analyze sentiment using VADER sentiment analyzer
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Sentiment label ('Positive', 'Negative', 'Neutral')
    """
    if not VADER_AVAILABLE:
        return analyze_sentiment_simple(text)
    
    if pd.isna(text) or text == "":
        return 'Neutral'
    
    scores = analyzer.polarity_scores(str(text))
    compound = scores['compound']
    
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def analyze_sentiment_textblob(text: str) -> str:
    """
    Analyze sentiment using TextBlob
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Sentiment label ('Positive', 'Negative', 'Neutral')
    """
    if not TEXTBLOB_AVAILABLE:
        return analyze_sentiment_simple(text)
    
    if pd.isna(text) or text == "":
        return 'Neutral'
    
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'


def combine_sentiments(sentiments: List[str]) -> str:
    """
    Combine multiple sentiment predictions using majority vote
    
    Args:
        sentiments (List[str]): List of sentiment predictions
        
    Returns:
        str: Final sentiment based on majority vote
    """
    sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
    return max(sentiment_counts, key=sentiment_counts.get)


def analyze_message_sentiment(text: str, method: str = 'combined') -> str:
    """
    Analyze sentiment of a single message
    
    Args:
        text (str): Message text to analyze
        method (str): Method to use ('simple', 'vader', 'textblob', 'combined')
        
    Returns:
        str: Sentiment label
    """
    if method == 'simple':
        return analyze_sentiment_simple(text)
    elif method == 'vader':
        return analyze_sentiment_vader(text)
    elif method == 'textblob':
        return analyze_sentiment_textblob(text)
    elif method == 'combined':
        sentiments = [
            analyze_sentiment_simple(text),
            analyze_sentiment_vader(text),
            analyze_sentiment_textblob(text)
        ]
        return combine_sentiments(sentiments)
    else:
        raise ValueError("Method must be one of: 'simple', 'vader', 'textblob', 'combined'")


def calculate_sentiment_score(sentiment: str) -> int:
    """
    Convert sentiment label to numerical score
    
    Args:
        sentiment (str): Sentiment label
        
    Returns:
        int: Numerical score (+1 for Positive, -1 for Negative, 0 for Neutral)
    """
    if sentiment == 'Positive':
        return 1
    elif sentiment == 'Negative':
        return -1
    else:  # Neutral
        return 0


def process_dataframe_sentiments(df: pd.DataFrame, 
                                text_column: str = 'combined_text',
                                method: str = 'combined') -> pd.DataFrame:
    """
    Process sentiment analysis for entire dataframe
    
    Args:
        df (pd.DataFrame): DataFrame containing text data
        text_column (str): Name of column containing text to analyze
        method (str): Sentiment analysis method to use
        
    Returns:
        pd.DataFrame: DataFrame with added sentiment columns
    """
    df_copy = df.copy()
    
    # Analyze sentiments
    df_copy['sentiment'] = df_copy[text_column].apply(
        lambda x: analyze_message_sentiment(x, method)
    )
    
    # Calculate numerical scores
    df_copy['sentiment_score'] = df_copy['sentiment'].apply(calculate_sentiment_score)
    
    return df_copy


def get_sentiment_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get sentiment distribution statistics
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment column
        
    Returns:
        Dict[str, Any]: Dictionary containing sentiment statistics
    """
    sentiment_counts = df['sentiment'].value_counts()
    total_messages = len(df)
    
    return {
        'total_messages': total_messages,
        'sentiment_counts': sentiment_counts.to_dict(),
        'sentiment_percentages': {
            sentiment: (count / total_messages) * 100 
            for sentiment, count in sentiment_counts.items()
        },
        'positive_ratio': sentiment_counts.get('Positive', 0) / total_messages,
        'negative_ratio': sentiment_counts.get('Negative', 0) / total_messages,
        'neutral_ratio': sentiment_counts.get('Neutral', 0) / total_messages
    }


if __name__ == "__main__":
    # Example usage
    print("Sentiment Analyzer Module")
    print("=" * 40)
    
    # Test sentiment analysis
    test_messages = [
        "I love this project! It's going great!",
        "This is terrible and I hate it",
        "The meeting is scheduled for tomorrow",
        "Excellent work everyone, well done!",
        "We have some serious problems to address"
    ]
    
    print("\nTesting sentiment analysis:")
    for message in test_messages:
        sentiment = analyze_message_sentiment(message, 'combined')
        score = calculate_sentiment_score(sentiment)
        print(f"Message: '{message}'")
        print(f"Sentiment: {sentiment} (Score: {score})")
        print("-" * 30)
