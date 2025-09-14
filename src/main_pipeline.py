"""
Main Analysis Pipeline
Orchestrates the complete employee sentiment analysis workflow
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import custom modules
from sentiment_analyzer import (
    preprocess_text, process_dataframe_sentiments, 
    get_sentiment_distribution
)
from employee_scoring import (
    calculate_monthly_scores, get_all_monthly_rankings,
    calculate_employee_statistics, get_performance_summary
)
from flight_risk_analyzer import (
    identify_flight_risk_employees, analyze_employee_risk_patterns,
    create_flight_risk_summary, generate_flight_risk_alerts
)
from data_visualizer import generate_all_visualizations


class SentimentAnalysisPipeline:
    """
    Complete pipeline for employee sentiment analysis
    """
    
    def __init__(self, data_path: str = "data/raw/test.csv"):
        """
        Initialize the pipeline
        
        Args:
            data_path (str): Path to the raw data file
        """
        self.data_path = data_path
        self.df = None
        self.monthly_scores = None
        self.flight_risk_df = None
        self.risk_analysis_df = None
        self.rankings_df = None
        
        # Create output directories
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("visualizations", exist_ok=True)
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess the raw data
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        print("📊 STEP 1: Loading and Preprocessing Data")
        print("-" * 50)
        
        try:
            # Load data
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded: {self.df.shape[0]} records, {self.df.shape[1]} columns")
            
            # Basic info
            print(f"Columns: {list(self.df.columns)}")
            print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
            print(f"Unique employees: {self.df['from'].nunique()}")
            
            # Preprocess text
            self.df['Subject_clean'] = self.df['Subject'].apply(preprocess_text)
            self.df['body_clean'] = self.df['body'].apply(preprocess_text)
            self.df['combined_text'] = self.df['Subject_clean'] + ' ' + self.df['body_clean']
            
            # Convert date
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            print("✅ Data preprocessing completed")
            return self.df
            
        except Exception as e:
            print(f"❌ Error in data loading: {e}")
            raise
    
    def analyze_sentiments(self) -> pd.DataFrame:
        """
        Perform sentiment analysis on the data
        
        Returns:
            pd.DataFrame: Data with sentiment labels
        """
        print("\n🎭 STEP 2: Sentiment Analysis")
        print("-" * 50)
        
        try:
            # Perform sentiment analysis
            self.df = process_dataframe_sentiments(self.df, 'combined_text', 'combined')
            
            # Get distribution
            sentiment_dist = get_sentiment_distribution(self.df)
            
            print("✅ Sentiment analysis completed")
            print("Sentiment Distribution:")
            for sentiment, count in sentiment_dist['sentiment_counts'].items():
                percentage = sentiment_dist['sentiment_percentages'][sentiment]
                print(f"  {sentiment}: {count} ({percentage:.1f}%)")
            
            # Save sentiment-labeled data
            output_path = "data/processed/sentiment_labeled_data.csv"
            self.df.to_csv(output_path, index=False)
            print(f"✅ Sentiment-labeled data saved to {output_path}")
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error in sentiment analysis: {e}")
            raise
    
    def calculate_employee_scores(self) -> pd.DataFrame:
        """
        Calculate monthly sentiment scores for employees
        
        Returns:
            pd.DataFrame: Monthly scores data
        """
        print("\n📈 STEP 3: Employee Score Calculation")
        print("-" * 50)
        
        try:
            # Calculate monthly scores
            self.monthly_scores = calculate_monthly_scores(self.df)
            
            print(f"✅ Monthly scores calculated for {len(self.monthly_scores)} employee-month combinations")
            print("Sample monthly scores:")
            print(self.monthly_scores.head(10).to_string(index=False))
            
            # Save monthly scores
            output_path = "data/processed/monthly_scores.csv"
            self.monthly_scores.to_csv(output_path, index=False)
            print(f"✅ Monthly scores saved to {output_path}")
            
            return self.monthly_scores
            
        except Exception as e:
            print(f"❌ Error in score calculation: {e}")
            raise
    
    def generate_employee_rankings(self) -> pd.DataFrame:
        """
        Generate employee rankings for all months
        
        Returns:
            pd.DataFrame: Rankings data
        """
        print("\n🏆 STEP 4: Employee Rankings")
        print("-" * 50)
        
        try:
            # Generate rankings
            self.rankings_df = get_all_monthly_rankings(self.monthly_scores)
            
            print(f"✅ Employee rankings calculated for {self.monthly_scores['year_month_str'].nunique()} months")
            print("Sample rankings:")
            print(self.rankings_df.head(10).to_string(index=False))
            
            # Save rankings
            output_path = "data/processed/employee_rankings.csv"
            self.rankings_df.to_csv(output_path, index=False)
            print(f"✅ Employee rankings saved to {output_path}")
            
            return self.rankings_df
            
        except Exception as e:
            print(f"❌ Error in ranking generation: {e}")
            raise
    
    def identify_flight_risks(self) -> pd.DataFrame:
        """
        Identify employees at flight risk
        
        Returns:
            pd.DataFrame: Flight risk data
        """
        print("\n⚠️  STEP 5: Flight Risk Identification")
        print("-" * 50)
        
        try:
            # Identify flight risk employees
            self.flight_risk_df = identify_flight_risk_employees(self.df)
            
            print(f"✅ Flight risk analysis completed")
            print(f"Flight risk employees identified: {len(self.flight_risk_df)}")
            
            if len(self.flight_risk_df) > 0:
                print("Flight risk employees:")
                print(self.flight_risk_df[['employee', 'negative_count', 'window_start', 'window_end']].to_string(index=False))
                
                # Generate alerts
                alerts = generate_flight_risk_alerts(self.flight_risk_df)
                print(f"\n🚨 ALERTS GENERATED:")
                for alert in alerts:
                    print(f"- {alert['alert_message']}")
                    print(f"  Recommendation: {alert['recommendation']}")
            else:
                print("No flight risk employees identified")
            
            # Save flight risk data
            output_path = "data/processed/flight_risk_employees.csv"
            self.flight_risk_df.to_csv(output_path, index=False)
            print(f"✅ Flight risk data saved to {output_path}")
            
            return self.flight_risk_df
            
        except Exception as e:
            print(f"❌ Error in flight risk identification: {e}")
            raise
    
    def analyze_risk_patterns(self) -> pd.DataFrame:
        """
        Analyze risk patterns for all employees
        
        Returns:
            pd.DataFrame: Risk analysis data
        """
        print("\n🔍 STEP 6: Risk Pattern Analysis")
        print("-" * 50)
        
        try:
            # Analyze risk patterns
            self.risk_analysis_df = analyze_employee_risk_patterns(self.df)
            
            print(f"✅ Risk pattern analysis completed for {len(self.risk_analysis_df)} employees")
            
            # Risk level summary
            risk_summary = self.risk_analysis_df['risk_level'].value_counts()
            print("Risk Level Distribution:")
            for level, count in risk_summary.items():
                print(f"  {level}: {count} employees")
            
            # Save risk analysis
            output_path = "data/processed/risk_analysis.csv"
            self.risk_analysis_df.to_csv(output_path, index=False)
            print(f"✅ Risk analysis saved to {output_path}")
            
            return self.risk_analysis_df
            
        except Exception as e:
            print(f"❌ Error in risk pattern analysis: {e}")
            raise
    
    def generate_visualizations(self) -> None:
        """
        Generate all visualizations
        """
        print("\n📊 STEP 7: Generating Visualizations")
        print("-" * 50)
        
        try:
            generate_all_visualizations(
                self.df, 
                self.monthly_scores, 
                self.flight_risk_df,
                self.risk_analysis_df,
                "visualizations"
            )
            print("✅ All visualizations generated successfully")
            
        except Exception as e:
            print(f"❌ Error in visualization generation: {e}")
            # Don't raise - visualizations are optional
    
    def generate_final_report(self) -> Dict:
        """
        Generate final summary report
        
        Returns:
            Dict: Complete analysis summary
        """
        print("\n📋 STEP 8: Generating Final Report")
        print("-" * 50)
        
        try:
            # Get performance summary
            performance_summary = get_performance_summary(self.df)
            
            # Get flight risk summary
            flight_risk_summary = create_flight_risk_summary(self.df)
            
            # Get employee statistics
            employee_stats = calculate_employee_statistics(self.df)
            
            # Create comprehensive report
            report = {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_summary': {
                    'total_employees': performance_summary['total_employees'],
                    'total_messages': performance_summary['total_messages'],
                    'date_range': performance_summary['date_range'],
                    'analysis_period_days': performance_summary['date_range']['duration_days']
                },
                'sentiment_analysis': {
                    'distribution': performance_summary['sentiment_distribution'],
                    'overall_score': performance_summary['overall_sentiment_score'],
                    'average_score': round(performance_summary['average_sentiment_score'], 3)
                },
                'employee_performance': {
                    'top_performer': performance_summary['top_performer'],
                    'most_active': performance_summary['most_active'],
                    'score_statistics': performance_summary['score_statistics']
                },
                'flight_risk_analysis': flight_risk_summary,
                'key_findings': self._generate_key_findings(),
                'recommendations': self._generate_recommendations()
            }
            
            # Save report
            import json
            with open('data/processed/final_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print("✅ Final report generated and saved")
            return report
            
        except Exception as e:
            print(f"❌ Error in report generation: {e}")
            raise
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from the analysis"""
        findings = []
        
        # Sentiment distribution finding
        sentiment_dist = self.df['sentiment'].value_counts()
        positive_pct = (sentiment_dist.get('Positive', 0) / len(self.df)) * 100
        negative_pct = (sentiment_dist.get('Negative', 0) / len(self.df)) * 100
        
        findings.append(f"Overall sentiment is {positive_pct:.1f}% positive, {negative_pct:.1f}% negative")
        
        # Flight risk finding
        if len(self.flight_risk_df) > 0:
            findings.append(f"{len(self.flight_risk_df)} employees identified as flight risk")
        else:
            findings.append("No employees currently at flight risk")
        
        # High performers
        top_employee = self.monthly_scores.groupby('employee')['total_score'].sum().idxmax()
        findings.append(f"Top performing employee: {top_employee.split('@')[0]}")
        
        # Activity levels
        avg_messages = len(self.df) / self.df['from'].nunique()
        findings.append(f"Average messages per employee: {avg_messages:.1f}")
        
        return findings
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Flight risk recommendations
        if len(self.flight_risk_df) > 0:
            recommendations.append("Schedule immediate check-ins with flight risk employees")
            recommendations.append("Implement stress management and support programs")
        
        # High negative sentiment recommendations
        high_negative_employees = len(self.risk_analysis_df[
            self.risk_analysis_df['negative_percentage'] > 20
        ])
        if high_negative_employees > 0:
            recommendations.append(f"Monitor {high_negative_employees} employees with high negative sentiment")
        
        # General recommendations
        recommendations.append("Implement regular sentiment monitoring system")
        recommendations.append("Create feedback channels to address employee concerns")
        recommendations.append("Recognize and reward top-performing employees")
        
        return recommendations
    
    def run_complete_pipeline(self) -> Dict:
        """
        Run the complete analysis pipeline
        
        Returns:
            Dict: Final analysis report
        """
        print("🚀 STARTING EMPLOYEE SENTIMENT ANALYSIS PIPELINE")
        print("=" * 60)
        
        try:
            # Execute all steps
            self.load_and_preprocess_data()
            self.analyze_sentiments()
            self.calculate_employee_scores()
            self.generate_employee_rankings()
            self.identify_flight_risks()
            self.analyze_risk_patterns()
            self.generate_visualizations()
            report = self.generate_final_report()
            
            print("\n" + "=" * 60)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            # Print summary
            self._print_final_summary(report)
            
            return report
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            raise
    
    def _print_final_summary(self, report: Dict) -> None:
        """Print final summary"""
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"Total Employees: {report['data_summary']['total_employees']}")
        print(f"Total Messages: {report['data_summary']['total_messages']}")
        print(f"Analysis Period: {report['data_summary']['date_range']['start']} to {report['data_summary']['date_range']['end']}")
        
        print(f"\n🎭 SENTIMENT BREAKDOWN:")
        for sentiment, count in report['sentiment_analysis']['distribution'].items():
            pct = (count / report['data_summary']['total_messages']) * 100
            print(f"  {sentiment}: {count} ({pct:.1f}%)")
        
        print(f"\n⚠️  FLIGHT RISK STATUS:")
        print(f"Flight Risk Employees: {report['flight_risk_analysis']['flight_risk_count']}")
        print(f"High Risk Employees: {report['flight_risk_analysis']['high_risk_employees']}")
        
        print(f"\n🏆 TOP PERFORMER:")
        top_perf = report['employee_performance']['top_performer']
        print(f"  {top_perf['employee'].split('@')[0]} (Score: {top_perf['total_score']})")
        
        print(f"\n💡 KEY FINDINGS:")
        for finding in report['key_findings']:
            print(f"  • {finding}")
        
        print(f"\n📋 FILES GENERATED:")
        print("  • data/processed/sentiment_labeled_data.csv")
        print("  • data/processed/monthly_scores.csv")
        print("  • data/processed/employee_rankings.csv")
        print("  • data/processed/flight_risk_employees.csv")
        print("  • data/processed/risk_analysis.csv")
        print("  • data/processed/final_report.json")
        print("  • visualizations/*.png (multiple charts)")


if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = SentimentAnalysisPipeline("data/raw/test.csv")
    
    try:
        final_report = pipeline.run_complete_pipeline()
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)
