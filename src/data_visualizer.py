"""
Data Visualization Module
Creates comprehensive visualizations for sentiment analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")


def setup_visualization_directory(base_path: str = "visualizations") -> str:
    """
    Setup directory for saving visualizations
    
    Args:
        base_path (str): Base directory path
        
    Returns:
        str: Full path to visualization directory
    """
    os.makedirs(base_path, exist_ok=True)
    return base_path


def plot_sentiment_distribution(df: pd.DataFrame, 
                               save_path: str = None,
                               figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Create pie chart and bar chart for sentiment distribution
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment column
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Pie chart
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax1.set_title('Sentiment Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
    ax2.set_title('Sentiment Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sentiment')
    ax2.set_ylabel('Number of Messages')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sentiment distribution plot saved to {save_path}")
    
    plt.show()


def plot_monthly_sentiment_trends(df: pd.DataFrame,
                                 save_path: str = None,
                                 figsize: Tuple[int, int] = (14, 8)) -> None:
    """
    Plot sentiment trends over time
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment and date columns
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Calculate monthly sentiment counts
    monthly_sentiment = df.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
    monthly_sentiment.index = monthly_sentiment.index.astype(str)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Stacked bar chart
    monthly_sentiment.plot(kind='bar', stacked=True, ax=ax1, 
                          color=['#e74c3c', '#f39c12', '#2ecc71'])
    ax1.set_title('Monthly Sentiment Distribution (Stacked)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Messages')
    ax1.legend(title='Sentiment')
    ax1.tick_params(axis='x', rotation=45)
    
    # Line chart for trends
    monthly_sentiment.plot(kind='line', ax=ax2, marker='o',
                          color=['#e74c3c', '#f39c12', '#2ecc71'])
    ax2.set_title('Monthly Sentiment Trends (Line Chart)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Number of Messages')
    ax2.legend(title='Sentiment')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Monthly sentiment trends plot saved to {save_path}")
    
    plt.show()


def plot_employee_sentiment_scores(monthly_scores: pd.DataFrame,
                                  top_n: int = 10,
                                  save_path: str = None,
                                  figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot employee sentiment scores
    
    Args:
        monthly_scores (pd.DataFrame): Monthly scores data
        top_n (int): Number of top/bottom employees to show
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size
    """
    # Calculate overall scores
    employee_totals = monthly_scores.groupby('employee')['total_score'].sum().sort_values()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Top performers
    top_employees = employee_totals.tail(top_n)
    bars1 = ax1.barh(range(len(top_employees)), top_employees.values, color='#2ecc71')
    ax1.set_yticks(range(len(top_employees)))
    ax1.set_yticklabels([emp.split('@')[0] for emp in top_employees.index])
    ax1.set_title(f'Top {top_n} Employees (Highest Scores)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Total Sentiment Score')
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                f'{int(width)}', ha='left', va='center')
    
    # Bottom performers
    bottom_employees = employee_totals.head(top_n)
    bars2 = ax2.barh(range(len(bottom_employees)), bottom_employees.values, color='#e74c3c')
    ax2.set_yticks(range(len(bottom_employees)))
    ax2.set_yticklabels([emp.split('@')[0] for emp in bottom_employees.index])
    ax2.set_title(f'Bottom {top_n} Employees (Lowest Scores)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Total Sentiment Score')
    
    # Add value labels
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                f'{int(width)}', ha='left', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Employee sentiment scores plot saved to {save_path}")
    
    plt.show()


def plot_flight_risk_analysis(flight_risk_df: pd.DataFrame,
                             risk_analysis_df: pd.DataFrame,
                             save_path: str = None,
                             figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Plot flight risk analysis results
    
    Args:
        flight_risk_df (pd.DataFrame): Flight risk employees data
        risk_analysis_df (pd.DataFrame): Risk analysis for all employees
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Flight risk count
    if len(flight_risk_df) > 0:
        flight_risk_counts = flight_risk_df['employee'].value_counts()
        bars1 = ax1.bar(range(len(flight_risk_counts)), flight_risk_counts.values, color='#e74c3c')
        ax1.set_xticks(range(len(flight_risk_counts)))
        ax1.set_xticklabels([emp.split('@')[0] for emp in flight_risk_counts.index], rotation=45)
        ax1.set_title('Flight Risk Employees', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Risk Incidents')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{int(height)}', ha='center', va='bottom')
    else:
        ax1.text(0.5, 0.5, 'No Flight Risk\nEmployees Identified', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Flight Risk Employees', fontsize=12, fontweight='bold')
    
    # 2. Risk level distribution
    risk_levels = risk_analysis_df['risk_level'].value_counts()
    colors = ['#e74c3c', '#f39c12', '#2ecc71']  # Red, Orange, Green
    ax2.pie(risk_levels.values, labels=risk_levels.index, autopct='%1.1f%%',
            colors=colors[:len(risk_levels)])
    ax2.set_title('Employee Risk Level Distribution', fontsize=12, fontweight='bold')
    
    # 3. Negative percentage vs Risk score
    ax3.scatter(risk_analysis_df['negative_percentage'], 
               risk_analysis_df['overall_risk_score'],
               c=risk_analysis_df['overall_risk_score'], 
               cmap='Reds', alpha=0.7)
    ax3.set_xlabel('Negative Message Percentage (%)')
    ax3.set_ylabel('Overall Risk Score')
    ax3.set_title('Risk Score vs Negative Messages', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Message activity vs sentiment
    ax4.scatter(risk_analysis_df['total_messages'],
               risk_analysis_df['overall_risk_score'],
               c=risk_analysis_df['negative_percentage'],
               cmap='Reds', alpha=0.7)
    ax4.set_xlabel('Total Messages')
    ax4.set_ylabel('Overall Risk Score')
    ax4.set_title('Activity Level vs Risk Score', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Flight risk analysis plot saved to {save_path}")
    
    plt.show()


def plot_employee_timeline(df: pd.DataFrame, 
                          employee_email: str,
                          save_path: str = None,
                          figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Plot individual employee sentiment timeline
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment data
        employee_email (str): Employee email to analyze
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size
    """
    employee_data = df[df['from'] == employee_email].copy()
    employee_data['date'] = pd.to_datetime(employee_data['date'])
    employee_data = employee_data.sort_values('date')
    
    if len(employee_data) == 0:
        print(f"No data found for employee: {employee_email}")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Timeline with sentiment colors
    colors = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        sentiment_data = employee_data[employee_data['sentiment'] == sentiment]
        if len(sentiment_data) > 0:
            ax1.scatter(sentiment_data['date'], sentiment_data['sentiment_score'],
                       c=colors[sentiment], label=sentiment, alpha=0.7, s=50)
    
    ax1.set_title(f'Sentiment Timeline: {employee_email.split("@")[0]}', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sentiment Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Cumulative sentiment score
    employee_data['cumulative_score'] = employee_data['sentiment_score'].cumsum()
    ax2.plot(employee_data['date'], employee_data['cumulative_score'], 
            color='#3498db', linewidth=2, marker='o', markersize=4)
    ax2.set_title('Cumulative Sentiment Score Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Score')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Employee timeline plot saved to {save_path}")
    
    plt.show()


def create_dashboard_summary(df: pd.DataFrame,
                           monthly_scores: pd.DataFrame,
                           flight_risk_df: pd.DataFrame,
                           save_path: str = None,
                           figsize: Tuple[int, int] = (16, 12)) -> None:
    """
    Create comprehensive dashboard with key metrics
    
    Args:
        df (pd.DataFrame): Main sentiment data
        monthly_scores (pd.DataFrame): Monthly scores data
        flight_risk_df (pd.DataFrame): Flight risk data
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Key metrics (top row)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    
    # Sentiment distribution (middle left)
    ax5 = fig.add_subplot(gs[1, :2])
    
    # Monthly trends (middle right)
    ax6 = fig.add_subplot(gs[1, 2:])
    
    # Employee rankings (bottom)
    ax7 = fig.add_subplot(gs[2, :])
    
    # Key metrics cards
    total_messages = len(df)
    total_employees = df['from'].nunique()
    flight_risk_count = len(flight_risk_df)
    avg_sentiment = df['sentiment_score'].mean()
    
    metrics = [
        (total_messages, "Total\nMessages", ax1),
        (total_employees, "Total\nEmployees", ax2),
        (flight_risk_count, "Flight Risk\nEmployees", ax3),
        (f"{avg_sentiment:.2f}", "Avg Sentiment\nScore", ax4)
    ]
    
    for value, label, ax in metrics:
        ax.text(0.5, 0.6, str(value), ha='center', va='center', 
               transform=ax.transAxes, fontsize=24, fontweight='bold')
        ax.text(0.5, 0.3, label, ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add border
        rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=2, 
                           edgecolor='gray', facecolor='lightgray', alpha=0.3)
        ax.add_patch(rect)
    
    # Sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax5.pie(sentiment_counts.values, labels=sentiment_counts.index, 
           autopct='%1.1f%%', colors=colors)
    ax5.set_title('Overall Sentiment Distribution', fontsize=12, fontweight='bold')
    
    # Monthly trends
    df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly_sentiment = df.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
    monthly_sentiment.index = monthly_sentiment.index.astype(str)
    monthly_sentiment.plot(kind='line', ax=ax6, marker='o',
                          color=['#e74c3c', '#f39c12', '#2ecc71'])
    ax6.set_title('Monthly Sentiment Trends', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Month')
    ax6.set_ylabel('Messages')
    ax6.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Top/Bottom employees
    employee_totals = monthly_scores.groupby('employee')['total_score'].sum().sort_values()
    top_5 = employee_totals.tail(5)
    bottom_5 = employee_totals.head(5)
    
    x_pos = list(range(len(top_5))) + [x + len(top_5) + 1 for x in range(len(bottom_5))]
    values = list(top_5.values) + list(bottom_5.values)
    labels = [emp.split('@')[0] for emp in top_5.index] + [emp.split('@')[0] for emp in bottom_5.index]
    colors_bar = ['#2ecc71'] * len(top_5) + ['#e74c3c'] * len(bottom_5)
    
    bars = ax7.bar(x_pos, values, color=colors_bar)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(labels, rotation=45, ha='right')
    ax7.set_title('Top 5 vs Bottom 5 Employees (Total Sentiment Score)', 
                 fontsize=12, fontweight='bold')
    ax7.set_ylabel('Total Sentiment Score')
    ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., 
                height + (abs(height) * 0.01 if height >= 0 else -abs(height) * 0.05),
                f'{int(height)}', ha='center', 
                va='bottom' if height >= 0 else 'top')
    
    # Add legend for top/bottom
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Top 5'),
                      Patch(facecolor='#e74c3c', label='Bottom 5')]
    ax7.legend(handles=legend_elements, loc='upper right')
    
    plt.suptitle('Employee Sentiment Analysis Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard summary saved to {save_path}")
    
    plt.show()


def generate_all_visualizations(df: pd.DataFrame,
                              monthly_scores: pd.DataFrame,
                              flight_risk_df: pd.DataFrame,
                              risk_analysis_df: pd.DataFrame,
                              output_dir: str = "visualizations") -> None:
    """
    Generate all visualizations and save them
    
    Args:
        df (pd.DataFrame): Main sentiment data
        monthly_scores (pd.DataFrame): Monthly scores data
        flight_risk_df (pd.DataFrame): Flight risk data
        risk_analysis_df (pd.DataFrame): Risk analysis data
        output_dir (str): Output directory for visualizations
    """
    setup_visualization_directory(output_dir)
    
    print("Generating all visualizations...")
    
    # 1. Sentiment distribution
    plot_sentiment_distribution(df, f"{output_dir}/01_sentiment_distribution.png")
    
    # 2. Monthly trends
    plot_monthly_sentiment_trends(df, f"{output_dir}/02_monthly_sentiment_trends.png")
    
    # 3. Employee scores
    plot_employee_sentiment_scores(monthly_scores, save_path=f"{output_dir}/03_employee_sentiment_scores.png")
    
    # 4. Flight risk analysis
    plot_flight_risk_analysis(flight_risk_df, risk_analysis_df, 
                             save_path=f"{output_dir}/04_flight_risk_analysis.png")
    
    # 5. Dashboard summary
    create_dashboard_summary(df, monthly_scores, flight_risk_df,
                            save_path=f"{output_dir}/05_dashboard_summary.png")
    
    # 6. Individual employee timelines (top 3 employees)
    employee_totals = monthly_scores.groupby('employee')['total_score'].sum().sort_values(ascending=False)
    top_employees = employee_totals.head(3).index
    
    for i, employee in enumerate(top_employees, 1):
        plot_employee_timeline(df, employee, 
                             save_path=f"{output_dir}/06_employee_timeline_{i}_{employee.split('@')[0]}.png")
    
    print(f"\nAll visualizations saved to {output_dir}/ directory")


if __name__ == "__main__":
    # Example usage
    print("Data Visualization Module")
    print("=" * 40)
    
    # Load sample data if available
    try:
        df = pd.read_csv('../data/processed/sentiment_labeled_data.csv')
        monthly_scores = pd.read_csv('../data/processed/monthly_scores.csv')
        
        print(f"Loaded {len(df)} records for visualization")
        
        # Create sample visualizations
        plot_sentiment_distribution(df)
        
    except FileNotFoundError:
        print("No processed data found. Run the main pipeline first.")
    except Exception as e:
        print(f"Error: {e}")
