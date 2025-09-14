"""
Flight Risk Analysis Module
Identifies employees at risk of leaving based on sentiment patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def identify_flight_risk_employees(df: pd.DataFrame, 
                                 negative_threshold: int = 4,
                                 window_days: int = 30) -> pd.DataFrame:
    """
    Identify employees at flight risk based on negative message patterns
    
    Flight risk criteria: 4+ negative messages in any 30-day rolling window
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment data and dates
        negative_threshold (int): Minimum negative messages to flag as flight risk
        window_days (int): Rolling window size in days
        
    Returns:
        pd.DataFrame: Flight risk employees with details
    """
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Get only negative messages
    negative_messages = df[df['sentiment'] == 'Negative'].copy()
    negative_messages = negative_messages.sort_values(['from', 'date'])
    
    flight_risk_employees = []
    
    print(f"Analyzing {len(negative_messages)} negative messages across {df['from'].nunique()} employees...")
    
    # For each employee, check rolling windows
    for employee in negative_messages['from'].unique():
        employee_negatives = negative_messages[negative_messages['from'] == employee].copy()
        
        if len(employee_negatives) < negative_threshold:
            continue
        
        # Check each negative message as potential start of window
        for i, (idx, message) in enumerate(employee_negatives.iterrows()):
            start_date = message['date']
            end_date = start_date + timedelta(days=window_days)
            
            # Count negative messages in this window
            window_negatives = employee_negatives[
                (employee_negatives['date'] >= start_date) & 
                (employee_negatives['date'] <= end_date)
            ]
            
            if len(window_negatives) >= negative_threshold:
                # Calculate additional risk metrics
                total_messages_in_window = df[
                    (df['from'] == employee) &
                    (df['date'] >= start_date) &
                    (df['date'] <= end_date)
                ]
                
                negative_ratio = len(window_negatives) / max(1, len(total_messages_in_window))
                
                flight_risk_employees.append({
                    'employee': employee,
                    'window_start': start_date.strftime('%Y-%m-%d'),
                    'window_end': end_date.strftime('%Y-%m-%d'),
                    'negative_count': len(window_negatives),
                    'total_messages_in_window': len(total_messages_in_window),
                    'negative_ratio': round(negative_ratio, 3),
                    'first_negative_date': window_negatives['date'].min().strftime('%Y-%m-%d'),
                    'last_negative_date': window_negatives['date'].max().strftime('%Y-%m-%d'),
                    'risk_score': len(window_negatives) * negative_ratio  # Custom risk metric
                })
                break  # Found flight risk for this employee, move to next
    
    flight_risk_df = pd.DataFrame(flight_risk_employees)
    
    if len(flight_risk_df) > 0:
        # Sort by risk score (highest risk first)
        flight_risk_df = flight_risk_df.sort_values('risk_score', ascending=False)
    
    return flight_risk_df


def analyze_employee_risk_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze risk patterns for all employees (not just flight risk)
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment data
        
    Returns:
        pd.DataFrame: Risk analysis for all employees
    """
    df['date'] = pd.to_datetime(df['date'])
    
    risk_analysis = []
    
    for employee in df['from'].unique():
        emp_data = df[df['from'] == employee].copy()
        emp_data = emp_data.sort_values('date')
        
        # Basic metrics
        total_messages = len(emp_data)
        negative_messages = len(emp_data[emp_data['sentiment'] == 'Negative'])
        positive_messages = len(emp_data[emp_data['sentiment'] == 'Positive'])
        neutral_messages = len(emp_data[emp_data['sentiment'] == 'Neutral'])
        
        # Calculate percentages
        negative_percentage = (negative_messages / total_messages) * 100
        positive_percentage = (positive_messages / total_messages) * 100
        
        # Calculate sentiment trend (recent vs early messages)
        if total_messages >= 10:
            recent_messages = emp_data.tail(min(10, total_messages // 2))
            early_messages = emp_data.head(min(10, total_messages // 2))
            
            recent_score = recent_messages['sentiment_score'].mean()
            early_score = early_messages['sentiment_score'].mean()
            sentiment_trend = recent_score - early_score
        else:
            sentiment_trend = 0
        
        # Check for consecutive negative periods
        negative_streaks = []
        current_streak = 0
        max_streak = 0
        
        for _, row in emp_data.iterrows():
            if row['sentiment'] == 'Negative':
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                if current_streak > 0:
                    negative_streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            negative_streaks.append(current_streak)
        
        # Calculate time since last positive message
        last_positive = emp_data[emp_data['sentiment'] == 'Positive']
        days_since_positive = None
        if len(last_positive) > 0:
            last_pos_date = last_positive['date'].max()
            days_since_positive = (df['date'].max() - last_pos_date).days
        
        # Calculate overall risk score
        risk_factors = {
            'high_negative_percentage': min(negative_percentage / 20, 1),  # Max 1 if >20% negative
            'negative_trend': max(0, -sentiment_trend),  # Positive if declining
            'long_negative_streak': min(max_streak / 5, 1),  # Max 1 if >5 consecutive
            'days_without_positive': min((days_since_positive or 0) / 60, 1) if days_since_positive else 0
        }
        
        overall_risk_score = sum(risk_factors.values()) / len(risk_factors)
        
        risk_analysis.append({
            'employee': employee,
            'total_messages': total_messages,
            'negative_count': negative_messages,
            'positive_count': positive_messages,
            'neutral_count': neutral_messages,
            'negative_percentage': round(negative_percentage, 2),
            'positive_percentage': round(positive_percentage, 2),
            'sentiment_trend': round(sentiment_trend, 3),
            'max_negative_streak': max_streak,
            'days_since_last_positive': days_since_positive,
            'overall_risk_score': round(overall_risk_score, 3),
            'risk_level': 'High' if overall_risk_score > 0.6 else 'Medium' if overall_risk_score > 0.3 else 'Low'
        })
    
    return pd.DataFrame(risk_analysis).sort_values('overall_risk_score', ascending=False)


def get_monthly_flight_risk_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze flight risk trends by month
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment data
        
    Returns:
        pd.DataFrame: Monthly flight risk metrics
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    
    monthly_trends = []
    
    for month in df['year_month'].unique():
        month_data = df[df['year_month'] == month]
        
        # Calculate flight risk employees for this month only
        month_df = month_data.copy()
        flight_risk = identify_flight_risk_employees(month_df)
        
        total_employees = month_data['from'].nunique()
        flight_risk_count = len(flight_risk)
        
        # Calculate other risk indicators
        high_negative_employees = len(
            month_data.groupby('from')['sentiment'].apply(
                lambda x: (x == 'Negative').sum() >= 3
            ).sum()
        )
        
        avg_sentiment_score = month_data['sentiment_score'].mean()
        negative_message_ratio = (month_data['sentiment'] == 'Negative').mean()
        
        monthly_trends.append({
            'month': str(month),
            'total_employees': total_employees,
            'flight_risk_employees': flight_risk_count,
            'flight_risk_percentage': round((flight_risk_count / total_employees) * 100, 2),
            'high_negative_employees': high_negative_employees,
            'avg_sentiment_score': round(avg_sentiment_score, 3),
            'negative_message_ratio': round(negative_message_ratio, 3),
            'total_messages': len(month_data)
        })
    
    return pd.DataFrame(monthly_trends)


def generate_flight_risk_alerts(flight_risk_df: pd.DataFrame) -> List[Dict]:
    """
    Generate actionable alerts for flight risk employees
    
    Args:
        flight_risk_df (pd.DataFrame): Flight risk employees data
        
    Returns:
        List[Dict]: Alert messages and recommendations
    """
    alerts = []
    
    for _, employee in flight_risk_df.iterrows():
        risk_level = "CRITICAL" if employee['negative_ratio'] > 0.5 else "HIGH"
        
        alert = {
            'employee': employee['employee'],
            'risk_level': risk_level,
            'negative_count': employee['negative_count'],
            'window_period': f"{employee['window_start']} to {employee['window_end']}",
            'negative_ratio': employee['negative_ratio'],
            'recommendation': get_risk_mitigation_recommendation(employee),
            'urgency': 'Immediate' if risk_level == "CRITICAL" else 'High',
            'alert_message': f"Employee {employee['employee']} sent {employee['negative_count']} negative messages "
                           f"between {employee['window_start']} and {employee['window_end']} "
                           f"({employee['negative_ratio']:.1%} of all messages in period)"
        }
        
        alerts.append(alert)
    
    return alerts


def get_risk_mitigation_recommendation(employee_data: Dict) -> str:
    """
    Generate personalized risk mitigation recommendations
    
    Args:
        employee_data (Dict): Employee flight risk data
        
    Returns:
        str: Recommendation text
    """
    negative_ratio = employee_data['negative_ratio']
    negative_count = employee_data['negative_count']
    
    if negative_ratio > 0.7:
        return ("URGENT: Schedule immediate one-on-one meeting. "
                "Consider workload assessment and stress management support.")
    elif negative_ratio > 0.5:
        return ("Schedule check-in meeting within 48 hours. "
                "Review recent projects and identify pain points.")
    elif negative_count >= 6:
        return ("Monitor closely and schedule informal check-in. "
                "Consider team dynamics assessment.")
    else:
        return ("Standard monitoring. Consider including in next regular review.")


def create_flight_risk_summary(df: pd.DataFrame) -> Dict:
    """
    Create comprehensive flight risk summary
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment data
        
    Returns:
        Dict: Summary of flight risk analysis
    """
    flight_risk_employees = identify_flight_risk_employees(df)
    risk_patterns = analyze_employee_risk_patterns(df)
    
    summary = {
        'total_employees': df['from'].nunique(),
        'flight_risk_count': len(flight_risk_employees),
        'flight_risk_percentage': round((len(flight_risk_employees) / df['from'].nunique()) * 100, 2),
        'high_risk_employees': len(risk_patterns[risk_patterns['risk_level'] == 'High']),
        'medium_risk_employees': len(risk_patterns[risk_patterns['risk_level'] == 'Medium']),
        'low_risk_employees': len(risk_patterns[risk_patterns['risk_level'] == 'Low']),
        'flight_risk_employees': flight_risk_employees[['employee', 'negative_count', 'window_start', 'window_end']].to_dict('records') if len(flight_risk_employees) > 0 else [],
        'recommendations': {
            'immediate_action': len(flight_risk_employees),
            'monitoring_required': len(risk_patterns[risk_patterns['overall_risk_score'] > 0.3]),
            'total_at_risk': len(flight_risk_employees) + len(risk_patterns[risk_patterns['risk_level'] == 'High'])
        }
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Flight Risk Analysis Module")
    print("=" * 40)
    
    # Load sample data if available
    try:
        df = pd.read_csv('../data/processed/sentiment_labeled_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"Loaded {len(df)} records for {df['from'].nunique()} employees")
        
        # Identify flight risk employees
        flight_risk = identify_flight_risk_employees(df)
        print(f"\nFlight Risk Employees Identified: {len(flight_risk)}")
        
        if len(flight_risk) > 0:
            print(flight_risk[['employee', 'negative_count', 'window_start', 'window_end']].to_string(index=False))
            
            # Generate alerts
            alerts = generate_flight_risk_alerts(flight_risk)
            print(f"\nGenerated {len(alerts)} alerts")
            for alert in alerts[:3]:  # Show first 3 alerts
                print(f"- {alert['alert_message']}")
                print(f"  Recommendation: {alert['recommendation']}")
        
        # Risk analysis summary
        summary = create_flight_risk_summary(df)
        print(f"\n--- FLIGHT RISK SUMMARY ---")
        print(f"Total Employees: {summary['total_employees']}")
        print(f"Flight Risk: {summary['flight_risk_count']} ({summary['flight_risk_percentage']}%)")
        print(f"High Risk: {summary['high_risk_employees']}")
        print(f"Medium Risk: {summary['medium_risk_employees']}")
        print(f"Low Risk: {summary['low_risk_employees']}")
        
    except FileNotFoundError:
        print("No processed data found. Run the main pipeline first.")
    except Exception as e:
        print(f"Error: {e}")
