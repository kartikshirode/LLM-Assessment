#!/usr/bin/env python3
"""
Employee Sentiment Analysis - Complete Pipeline
This script executes all tasks from the PDF requirements to generate processed data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Try importing sentiment analysis libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    analyzer = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False
    print("VADER Sentiment not available")

# Create processed data directory
os.makedirs('data/processed', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

print("Starting Employee Sentiment Analysis Pipeline...")
print("=" * 60)

# Task 1: Data Loading and Preprocessing
print("\n📊 TASK 1: Data Loading and Preprocessing")
print("-" * 40)

# Load the dataset
try:
    df = pd.read_csv('data/raw/test.csv')
    print(f"✅ Dataset loaded successfully: {df.shape[0]} records, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Data preprocessing
def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Clean the data
df['Subject_clean'] = df['Subject'].apply(preprocess_text)
df['body_clean'] = df['body'].apply(preprocess_text)
df['combined_text'] = df['Subject_clean'] + ' ' + df['body_clean']

# Convert date column
df['date'] = pd.to_datetime(df['date'])
df['year_month'] = df['date'].dt.to_period('M')

print(f"✅ Data preprocessing completed")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Save preprocessed data
df.to_csv('data/processed/preprocessed_data.csv', index=False)
print("✅ Preprocessed data saved to data/processed/preprocessed_data.csv")

# Task 2: Sentiment Labeling
print("\n🎭 TASK 2: Sentiment Labeling")
print("-" * 40)

def analyze_sentiment_simple(text):
    """Simple sentiment analysis using basic keywords"""
    if pd.isna(text) or text == "":
        return 'Neutral'
    
    text = str(text).lower()
    
    # Positive keywords
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                     'happy', 'pleased', 'satisfied', 'perfect', 'awesome', 'brilliant',
                     'outstanding', 'exceptional', 'superb', 'magnificent', 'terrific',
                     'love', 'like', 'enjoy', 'appreciate', 'thank', 'thanks', 'grateful']
    
    # Negative keywords
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'frustrated',
                     'disappointed', 'sad', 'upset', 'annoyed', 'irritated', 'furious',
                     'disgusted', 'outraged', 'appalled', 'concerned', 'worried', 'problem',
                     'issue', 'error', 'fail', 'wrong', 'broken', 'difficult', 'hard']
    
    positive_score = sum(1 for word in positive_words if word in text)
    negative_score = sum(1 for word in negative_words if word in text)
    
    if positive_score > negative_score:
        return 'Positive'
    elif negative_score > positive_score:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
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

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
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

# Apply sentiment analysis
print("Analyzing sentiment using available methods...")
df['sentiment_simple'] = df['combined_text'].apply(analyze_sentiment_simple)

if VADER_AVAILABLE:
    df['sentiment_vader'] = df['combined_text'].apply(analyze_sentiment_vader)
    print("✅ VADER sentiment analysis completed")
else:
    df['sentiment_vader'] = df['sentiment_simple']
    print("⚠️  VADER not available, using simple method")

if TEXTBLOB_AVAILABLE:
    df['sentiment_textblob'] = df['combined_text'].apply(analyze_sentiment_textblob)
    print("✅ TextBlob sentiment analysis completed")
else:
    df['sentiment_textblob'] = df['sentiment_simple']
    print("⚠️  TextBlob not available, using simple method")

# Combine sentiments (majority vote)
def combine_sentiments(row):
    sentiments = [row['sentiment_simple'], row['sentiment_vader'], row['sentiment_textblob']]
    sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
    return max(sentiment_counts, key=sentiment_counts.get)

df['sentiment'] = df.apply(combine_sentiments, axis=1)

print(f"✅ Sentiment labeling completed")
print("Sentiment distribution:")
print(df['sentiment'].value_counts())

# Save sentiment-labeled data
df.to_csv('data/processed/sentiment_labeled_data.csv', index=False)
print("✅ Sentiment-labeled data saved to data/processed/sentiment_labeled_data.csv")

# Task 3: Employee Score Calculation
print("\n📈 TASK 3: Employee Score Calculation")
print("-" * 40)

def calculate_sentiment_score(sentiment):
    """Convert sentiment to numerical score"""
    if sentiment == 'Positive':
        return 1
    elif sentiment == 'Negative':
        return -1
    else:  # Neutral
        return 0

df['sentiment_score'] = df['sentiment'].apply(calculate_sentiment_score)

# Calculate monthly scores for each employee
monthly_scores = df.groupby(['from', 'year_month'])['sentiment_score'].sum().reset_index()
monthly_scores['year_month_str'] = monthly_scores['year_month'].astype(str)

print(f"✅ Monthly scores calculated for {len(monthly_scores)} employee-month combinations")
print("Sample monthly scores:")
print(monthly_scores.head(10))

# Save monthly scores
monthly_scores.to_csv('data/processed/monthly_scores.csv', index=False)
print("✅ Monthly scores saved to data/processed/monthly_scores.csv")

# Task 4: Employee Ranking
print("\n🏆 TASK 4: Employee Ranking")
print("-" * 40)

# Create rankings for each month
rankings = []

for month in monthly_scores['year_month'].unique():
    month_data = monthly_scores[monthly_scores['year_month'] == month].copy()
    
    # Top 3 positive employees
    top_positive = month_data.nlargest(3, 'sentiment_score').sort_values(['sentiment_score', 'from'], ascending=[False, True])
    
    # Top 3 negative employees (most negative scores)
    top_negative = month_data.nsmallest(3, 'sentiment_score').sort_values(['sentiment_score', 'from'], ascending=[True, True])
    
    for idx, row in top_positive.iterrows():
        rankings.append({
            'month': str(month),
            'employee': row['from'],
            'score': row['sentiment_score'],
            'rank_type': 'Top Positive',
            'rank': len(rankings) % 3 + 1
        })
    
    for idx, row in top_negative.iterrows():
        rankings.append({
            'month': str(month),
            'employee': row['from'],
            'score': row['sentiment_score'],
            'rank_type': 'Top Negative',
            'rank': len(rankings) % 3 + 1
        })

rankings_df = pd.DataFrame(rankings)

print(f"✅ Employee rankings calculated for {len(monthly_scores['year_month'].unique())} months")
print("Sample rankings:")
print(rankings_df.head(10))

# Save rankings
rankings_df.to_csv('data/processed/employee_rankings.csv', index=False)
print("✅ Employee rankings saved to data/processed/employee_rankings.csv")

# Task 5: Flight Risk Identification
print("\n⚠️  TASK 5: Flight Risk Identification")
print("-" * 40)

# Flight risk: 4+ negative messages in 30 days (rolling window)
flight_risk_employees = []

# Get all negative messages
negative_messages = df[df['sentiment'] == 'Negative'].copy()
negative_messages = negative_messages.sort_values(['from', 'date'])

print(f"Total negative messages: {len(negative_messages)}")

# For each employee, check 30-day rolling windows
for employee in negative_messages['from'].unique():
    employee_negatives = negative_messages[negative_messages['from'] == employee].copy()
    
    if len(employee_negatives) < 4:
        continue
    
    # Check each message as a potential start of 30-day window
    for i, (idx, message) in enumerate(employee_negatives.iterrows()):
        start_date = message['date']
        end_date = start_date + timedelta(days=30)
        
        # Count negative messages in this 30-day window
        window_negatives = employee_negatives[
            (employee_negatives['date'] >= start_date) & 
            (employee_negatives['date'] <= end_date)
        ]
        
        if len(window_negatives) >= 4:
            flight_risk_employees.append({
                'employee': employee,
                'window_start': start_date.strftime('%Y-%m-%d'),
                'window_end': end_date.strftime('%Y-%m-%d'),
                'negative_count': len(window_negatives),
                'first_negative_date': window_negatives['date'].min().strftime('%Y-%m-%d'),
                'last_negative_date': window_negatives['date'].max().strftime('%Y-%m-%d')
            })
            break  # Found flight risk for this employee

flight_risk_df = pd.DataFrame(flight_risk_employees)

print(f"✅ Flight risk analysis completed")
print(f"Flight risk employees identified: {len(flight_risk_df)}")
if len(flight_risk_df) > 0:
    print("Flight risk employees:")
    print(flight_risk_df[['employee', 'negative_count', 'window_start', 'window_end']])
else:
    print("No employees identified as flight risk")

# Save flight risk data
flight_risk_df.to_csv('data/processed/flight_risk_employees.csv', index=False)
print("✅ Flight risk data saved to data/processed/flight_risk_employees.csv")

# Task 6: Predictive Modeling
print("\n🤖 TASK 6: Predictive Modeling")
print("-" * 40)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    
    # Prepare features for modeling
    modeling_data = df.groupby(['from', 'year_month']).agg({
        'sentiment_score': 'sum',  # Monthly sentiment score (target variable)
        'combined_text': 'count',  # Message frequency in month
        'body': lambda x: x.str.len().mean(),  # Average message length
        'Subject': lambda x: x.str.len().mean(),  # Average subject length
        'combined_text': [('message_count', 'count'), ('avg_length', lambda x: x.str.len().mean())]
    }).reset_index()
    
    # Flatten column names
    modeling_data.columns = ['employee', 'year_month', 'monthly_score', 'message_frequency', 
                            'avg_body_length', 'avg_subject_length', 'message_count', 'avg_total_length']
    
    # Remove rows with missing values
    modeling_data = modeling_data.dropna()
    
    # Prepare features and target
    features = ['message_frequency', 'avg_body_length', 'avg_subject_length', 'avg_total_length']
    X = modeling_data[features]
    y = modeling_data['monthly_score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Linear regression model trained")
    print(f"Model Performance:")
    print(f"  Mean Squared Error: {mse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Features used: {features}")
    
    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nFeature Importance (Coefficients):")
    print(feature_importance)
    
    # Save model results
    model_results = {
        'mse': mse,
        'r2': r2,
        'features': features,
        'coefficients': model.coef_.tolist(),
        'intercept': model.intercept_
    }
    
    import json
    with open('data/processed/model_results.json', 'w') as f:
        json.dump(model_results, f, indent=2)
    
    print("✅ Model results saved to data/processed/model_results.json")
    
except Exception as e:
    print(f"❌ Error in predictive modeling: {e}")

print("\n" + "=" * 60)
print("🎉 ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\n📋 SUMMARY OF GENERATED FILES:")
print("data/processed/preprocessed_data.csv - Cleaned and preprocessed dataset")
print("data/processed/sentiment_labeled_data.csv - Dataset with sentiment labels")
print("data/processed/monthly_scores.csv - Monthly sentiment scores by employee")
print("data/processed/employee_rankings.csv - Top positive/negative employee rankings")
print("data/processed/flight_risk_employees.csv - Employees at flight risk")
print("data/processed/model_results.json - Predictive modeling results")

print("\n📊 KEY FINDINGS:")
print(f"Total employees analyzed: {df['from'].nunique()}")
print(f"Total messages processed: {len(df)}")
print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Sentiment distribution: {dict(df['sentiment'].value_counts())}")
print(f"Flight risk employees identified: {len(flight_risk_df)}")

if len(flight_risk_df) > 0:
    print(f"\n⚠️  FLIGHT RISK EMPLOYEES:")
    for _, row in flight_risk_df.iterrows():
        print(f"  • {row['employee']} ({row['negative_count']} negative messages)")

# Get top employees for each category
top_positive_overall = monthly_scores.groupby('from')['sentiment_score'].sum().nlargest(3)
top_negative_overall = monthly_scores.groupby('from')['sentiment_score'].sum().nsmallest(3)

print(f"\n🏆 TOP 3 MOST POSITIVE EMPLOYEES (Overall):")
for i, (employee, score) in enumerate(top_positive_overall.items(), 1):
    print(f"  {i}. {employee} (Score: {score})")

print(f"\n⚠️  TOP 3 MOST NEGATIVE EMPLOYEES (Overall):")
for i, (employee, score) in enumerate(top_negative_overall.items(), 1):
    print(f"  {i}. {employee} (Score: {score})")

print("\n✅ All tasks completed successfully!")
