"""
Employee Scoring Module
Provides functions for calculating and managing employee sentiment scores
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_monthly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly sentiment scores for each employee
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment scores, employees, and dates
        
    Returns:
        pd.DataFrame: Monthly scores by employee
    """
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
    
    # Group by employee and month, sum sentiment scores
    monthly_scores = df.groupby(['from', 'year_month'])['sentiment_score'].agg([
        'sum',
        'count',
        'mean'
    ]).reset_index()
    
    monthly_scores.columns = ['employee', 'year_month', 'total_score', 'message_count', 'avg_score']
    monthly_scores['year_month_str'] = monthly_scores['year_month'].astype(str)
    
    return monthly_scores


def get_employee_rankings(monthly_scores: pd.DataFrame, 
                         month: str = None, 
                         top_n: int = 3) -> Dict[str, pd.DataFrame]:
    """
    Generate employee rankings for a specific month or overall
    
    Args:
        monthly_scores (pd.DataFrame): Monthly scores data
        month (str, optional): Specific month to rank (e.g., '2010-01')
        top_n (int): Number of top employees to return
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with top positive and negative employees
    """
    if month:
        # Filter for specific month
        month_data = monthly_scores[
            monthly_scores['year_month_str'] == month
        ].copy()
    else:
        # Overall rankings (sum across all months)
        month_data = monthly_scores.groupby('employee').agg({
            'total_score': 'sum',
            'message_count': 'sum',
            'avg_score': 'mean'
        }).reset_index()
    
    # Top positive employees (highest scores)
    top_positive = month_data.nlargest(top_n, 'total_score').sort_values(
        ['total_score', 'employee'], ascending=[False, True]
    )
    
    # Top negative employees (lowest scores)
    top_negative = month_data.nsmallest(top_n, 'total_score').sort_values(
        ['total_score', 'employee'], ascending=[True, True]
    )
    
    return {
        'top_positive': top_positive,
        'top_negative': top_negative
    }


def get_all_monthly_rankings(monthly_scores: pd.DataFrame, 
                           top_n: int = 3) -> pd.DataFrame:
    """
    Generate rankings for all months
    
    Args:
        monthly_scores (pd.DataFrame): Monthly scores data
        top_n (int): Number of top employees per category
        
    Returns:
        pd.DataFrame: All monthly rankings
    """
    all_rankings = []
    
    for month in monthly_scores['year_month_str'].unique():
        rankings = get_employee_rankings(monthly_scores, month, top_n)
        
        # Process top positive
        for idx, (_, row) in enumerate(rankings['top_positive'].iterrows()):
            all_rankings.append({
                'month': month,
                'employee': row['employee'],
                'score': row['total_score'],
                'message_count': row['message_count'],
                'rank_type': 'Top Positive',
                'rank': idx + 1
            })
        
        # Process top negative
        for idx, (_, row) in enumerate(rankings['top_negative'].iterrows()):
            all_rankings.append({
                'month': month,
                'employee': row['employee'],
                'score': row['total_score'],
                'message_count': row['message_count'],
                'rank_type': 'Top Negative',
                'rank': idx + 1
            })
    
    return pd.DataFrame(all_rankings)


def calculate_employee_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive statistics for each employee
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment data
        
    Returns:
        pd.DataFrame: Employee statistics
    """
    stats = df.groupby('from').agg({
        'sentiment_score': ['sum', 'mean', 'std', 'count'],
        'sentiment': lambda x: (x == 'Positive').sum(),
        'body': lambda x: x.str.len().mean()
    }).reset_index()
    
    # Flatten column names
    stats.columns = [
        'employee', 'total_score', 'avg_score', 'score_std', 'message_count',
        'positive_count', 'avg_message_length'
    ]
    
    # Add additional metrics
    negative_counts = df[df['sentiment'] == 'Negative'].groupby('from').size()
    neutral_counts = df[df['sentiment'] == 'Neutral'].groupby('from').size()
    
    stats['negative_count'] = stats['employee'].map(negative_counts).fillna(0)
    stats['neutral_count'] = stats['employee'].map(neutral_counts).fillna(0)
    
    # Calculate percentages
    stats['positive_percentage'] = (stats['positive_count'] / stats['message_count']) * 100
    stats['negative_percentage'] = (stats['negative_count'] / stats['message_count']) * 100
    stats['neutral_percentage'] = (stats['neutral_count'] / stats['message_count']) * 100
    
    # Calculate engagement score (messages per month)
    date_range = df['date'].max() - df['date'].min()
    months_active = max(1, date_range.days / 30.44)  # Average days per month
    stats['messages_per_month'] = stats['message_count'] / months_active
    
    return stats.sort_values('total_score', ascending=False)


def get_score_trends(monthly_scores: pd.DataFrame, 
                    employee: str = None) -> pd.DataFrame:
    """
    Get sentiment score trends over time
    
    Args:
        monthly_scores (pd.DataFrame): Monthly scores data
        employee (str, optional): Specific employee to analyze
        
    Returns:
        pd.DataFrame: Score trends data
    """
    if employee:
        trends = monthly_scores[
            monthly_scores['employee'] == employee
        ].copy().sort_values('year_month')
    else:
        # Overall trends (average across all employees)
        trends = monthly_scores.groupby('year_month_str').agg({
            'total_score': 'mean',
            'message_count': 'mean',
            'avg_score': 'mean'
        }).reset_index()
        trends['employee'] = 'Overall Average'
    
    return trends


def identify_score_anomalies(monthly_scores: pd.DataFrame, 
                           threshold: float = 2.0) -> pd.DataFrame:
    """
    Identify employees with unusual score patterns (outliers)
    
    Args:
        monthly_scores (pd.DataFrame): Monthly scores data
        threshold (float): Number of standard deviations for outlier detection
        
    Returns:
        pd.DataFrame: Employees with anomalous patterns
    """
    anomalies = []
    
    for employee in monthly_scores['employee'].unique():
        emp_data = monthly_scores[monthly_scores['employee'] == employee]
        
        if len(emp_data) < 3:  # Need at least 3 months of data
            continue
        
        scores = emp_data['total_score']
        mean_score = scores.mean()
        std_score = scores.std()
        
        if std_score == 0:  # No variation
            continue
        
        # Find months with extreme scores
        z_scores = np.abs((scores - mean_score) / std_score)
        extreme_months = emp_data[z_scores > threshold]
        
        for _, month_data in extreme_months.iterrows():
            anomalies.append({
                'employee': employee,
                'month': month_data['year_month_str'],
                'score': month_data['total_score'],
                'z_score': ((month_data['total_score'] - mean_score) / std_score),
                'employee_avg': mean_score,
                'deviation_type': 'High' if month_data['total_score'] > mean_score else 'Low'
            })
    
    return pd.DataFrame(anomalies)


def get_performance_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get overall performance summary across all employees
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment data
        
    Returns:
        Dict: Performance summary statistics
    """
    monthly_scores = calculate_monthly_scores(df)
    employee_stats = calculate_employee_statistics(df)
    
    summary = {
        'total_employees': df['from'].nunique(),
        'total_messages': len(df),
        'date_range': {
            'start': df['date'].min().strftime('%Y-%m-%d'),
            'end': df['date'].max().strftime('%Y-%m-%d'),
            'duration_days': (df['date'].max() - df['date'].min()).days
        },
        'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
        'overall_sentiment_score': df['sentiment_score'].sum(),
        'average_sentiment_score': df['sentiment_score'].mean(),
        'top_performer': {
            'employee': employee_stats.iloc[0]['employee'],
            'total_score': employee_stats.iloc[0]['total_score'],
            'message_count': employee_stats.iloc[0]['message_count']
        },
        'most_active': {
            'employee': employee_stats.loc[employee_stats['message_count'].idxmax(), 'employee'],
            'message_count': employee_stats['message_count'].max()
        },
        'score_statistics': {
            'mean': float(monthly_scores['total_score'].mean()),
            'median': float(monthly_scores['total_score'].median()),
            'std': float(monthly_scores['total_score'].std()),
            'min': float(monthly_scores['total_score'].min()),
            'max': float(monthly_scores['total_score'].max())
        }
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Employee Scoring Module")
    print("=" * 40)
    
    # Load sample data if available
    try:
        df = pd.read_csv('../data/processed/sentiment_labeled_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"Loaded {len(df)} records for {df['from'].nunique()} employees")
        
        # Calculate monthly scores
        monthly_scores = calculate_monthly_scores(df)
        print(f"\nCalculated monthly scores for {len(monthly_scores)} employee-month combinations")
        
        # Get overall rankings
        rankings = get_employee_rankings(monthly_scores)
        print(f"\nTop 3 Positive Employees (Overall):")
        print(rankings['top_positive'][['employee', 'total_score']].to_string(index=False))
        
        print(f"\nTop 3 Negative Employees (Overall):")
        print(rankings['top_negative'][['employee', 'total_score']].to_string(index=False))
        
    except FileNotFoundError:
        print("No processed data found. Run the main pipeline first.")
    except Exception as e:
        print(f"Error: {e}")
