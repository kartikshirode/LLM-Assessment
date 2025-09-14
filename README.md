# Employee Sentiment Analysis Project

## 📌 Project Overview

This project analyzes an unlabeled dataset of employee messages to assess sentiment and engagement using natural language processing (NLP) and statistical analysis techniques. The project implements six distinct tasks focusing on different aspects of data analysis and model development.

## 🎯 Project Objectives

The main goals are to evaluate employee sentiment and engagement by performing:

1. **Sentiment Labeling**: Automatically label each message as Positive, Negative, or Neutral
2. **Exploratory Data Analysis (EDA)**: Analyze and visualize data structure and trends
3. **Employee Score Calculation**: Compute monthly sentiment scores for each employee
4. **Employee Ranking**: Identify and rank employees by sentiment scores
5. **Flight Risk Identification**: Identify employees with 4+ negative messages in 30 days
6. **Predictive Modeling**: Develop linear regression model to analyze sentiment trends

## 🛠️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd employee-sentiment-analysis
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebooks**
   Execute notebooks in order from 01 to 07 for complete analysis

## 📊 Project Tasks

### Task 1: Sentiment Labeling
- **Notebook**: `02_Task1_Sentiment_Labeling.ipynb`
- **Objective**: Label each employee message as Positive, Negative, or Neutral
- **Approach**: Uses TextBlob and VADER sentiment analysis for robust classification

### Task 2: Exploratory Data Analysis
- **Notebook**: `03_Task2_Exploratory_Data_Analysis.ipynb`
- **Objective**: Understanding data structure, distribution, and trends
- **Includes**: Data structure analysis, sentiment distribution, time trends, patterns

### Task 3: Employee Score Calculation
- **Notebook**: `04_Task3_Employee_Score_Calculation.ipynb`
- **Objective**: Compute monthly sentiment scores for each employee
- **Scoring**: Positive (+1), Negative (-1), Neutral (0)

### Task 4: Employee Ranking
- **Notebook**: `05_Task4_Employee_Ranking.ipynb`
- **Objective**: Rank employees based on monthly sentiment scores
- **Output**: Top 3 positive and top 3 negative employees per month

### Task 5: Flight Risk Identification
- **Notebook**: `06_Task5_Flight_Risk_Identification.ipynb`
- **Objective**: Identify employees at risk of leaving
- **Criteria**: 4+ negative messages in any 30-day rolling period

### Task 6: Predictive Modeling
- **Notebook**: `07_Task6_Predictive_Modeling.ipynb`
- **Objective**: Develop linear regression model for sentiment trend analysis
- **Features**: Message frequency, length, word count, etc.

## � Project Structure

```
├── data/
│   ├── raw/                  # Original test.csv dataset
│   └── processed/            # Processed data files
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Task1_Sentiment_Labeling.ipynb
│   ├── 03_Task2_Exploratory_Data_Analysis.ipynb
│   ├── 04_Task3_Employee_Score_Calculation.ipynb
│   ├── 05_Task4_Employee_Ranking.ipynb
│   ├── 06_Task5_Flight_Risk_Identification.ipynb
│   └── 07_Task6_Predictive_Modeling.ipynb
├── visualizations/           # Charts and graphs
├── src/                     # Supporting Python modules
├── requirements.txt
└── README.md
```

## 📈 Key Findings

### Top Three Positive Employees (Latest Month)
*[To be updated after analysis]*
1. Employee Name - Score
2. Employee Name - Score  
3. Employee Name - Score

### Top Three Negative Employees (Latest Month)
*[To be updated after analysis]*
1. Employee Name - Score
2. Employee Name - Score
3. Employee Name - Score

### Flight Risk Employees
*[To be updated after analysis]*
- List of employees with 4+ negative messages in 30-day period

### Key Insights
*[To be updated after analysis]*
- Summary of main findings
- Recommendations for employee engagement
- Sentiment trends and patterns

## 🔧 Technologies Used

- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **TextBlob & NLTK**: Sentiment analysis
- **Scikit-learn**: Machine learning models
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualizations

## 📊 Visualizations

All charts and graphs are saved in the `visualizations/` folder:
- Sentiment distribution charts
- Employee ranking visualizations
- Flight risk analysis plots
- Model performance metrics

## 🎯 Model Performance

*[To be updated after model training]*
- Linear regression model metrics
- Feature importance analysis
- Prediction accuracy results

## � Contact

For questions about this analysis, please contact the project team.

---

*This project is part of an employee sentiment analysis evaluation focusing on NLP techniques and statistical modeling.*
