# ==============================================================================
# Project: IBM HR Analytics Employee Attrition & Performance
#
# Objective:
# To perform a comprehensive data analysis to uncover key factors that lead to
# employee attrition. The analysis will explore the relationship between the
# 'Attrition' variable and other features in the dataset, such as demographics,
# job details, and work-life balance.
#
# Dataset:
# The dataset is a fictional HR dataset created by IBM data scientists. It
# contains various employee-related features and an 'Attrition' column indicating
# whether an employee left the company.
# ==============================================================================

# 1. Importing necessary libraries
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# 2. Data Loading and Initial Exploration
# ------------------------------------------------------------------------------
# Load the dataset from the provided CSV file.
try:
    df = pd.read_csv(r"c:\users\Mohit\Documents\Unified Mentor Projects\IBM\WA_Fn-UseC_-HR-Employee-Attrition .csv")
    print("Dataset loaded successfully.\n")
except FileNotFoundError:
    print("Error: The CSV file 'WA_Fn-UseC_-HR-Employee-Attrition .csv' was not found.")
    print("Please make sure the file is in the same directory as the script.")
    exit()

# Display the shape of the dataset (rows, columns)
print("Shape of the dataset (rows, columns):", df.shape)

# Display the first 5 rows to get a quick overview
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Get a concise summary of the DataFrame, including data types and non-null values
print("\nDataset information:")
df.info()

# Check for missing values in each column
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Check for any duplicate rows
print("\nNumber of duplicate rows:", df.duplicated().sum())

# 3. Data Cleaning and Preprocessing
# ------------------------------------------------------------------------------
# The PDF analysis shows that certain columns have only a single unique value
# and thus provide no analytical value. We will drop these.
columns_to_drop = ['EmployeeCount', 'StandardHours', 'Over18']

for col in columns_to_drop:
    if col in df.columns and df[col].nunique() == 1:
        df.drop(columns=[col], inplace=True)
        print(f"\nDropped column '{col}' as it contains only one unique value.")

# As per the PDF, convert numerical categorical variables to more descriptive strings
# for better readability in analysis.
education_map = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
environment_satisfaction_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
job_involvement_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
job_satisfaction_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
performance_rating_map = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}
relationship_satisfaction_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
work_life_balance_map = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}

df['Education'] = df['Education'].map(education_map)
df['EnvironmentSatisfaction'] = df['EnvironmentSatisfaction'].map(environment_satisfaction_map)
df['JobInvolvement'] = df['JobInvolvement'].map(job_involvement_map)
df['JobSatisfaction'] = df['JobSatisfaction'].map(job_satisfaction_map)
df['PerformanceRating'] = df['PerformanceRating'].map(performance_rating_map)
df['RelationshipSatisfaction'] = df['RelationshipSatisfaction'].map(relationship_satisfaction_map)
df['WorkLifeBalance'] = df['WorkLifeBalance'].map(work_life_balance_map)

print("\nCategorical columns mapped to descriptive labels.")
print("\nUpdated DataFrame info:")
df.info()

# 4. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
# Analyze the Attrition variable
attrition_counts = df['Attrition'].value_counts()
attrition_rate = (attrition_counts['Yes'] / len(df)) * 100
print(f"\nOverall Attrition Rate: {attrition_rate:.2f}%")

# Visualize attrition by key categorical features
categorical_features = [
    'Gender', 'MaritalStatus', 'Department', 'JobRole', 'EducationField',
    'BusinessTravel', 'JobSatisfaction', 'EnvironmentSatisfaction',
    'WorkLifeBalance', 'JobInvolvement'
]

plt.figure(figsize=(18, 20))
plt.subplots_adjust(hspace=0.5)

for i, col in enumerate(categorical_features, 1):
    plt.subplot(5, 2, i)
    ax = sns.countplot(data=df, x=col, hue='Attrition', palette='Pastel1')
    plt.title(f'Attrition by {col}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Employees')
    plt.xlabel('')
    ax.legend(title='Attrition', loc='upper right')

plt.tight_layout()
plt.show()

# Visualize attrition by key numerical features
numerical_features = [
    'Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome',
    'TotalWorkingYears'
]

plt.figure(figsize=(18, 15))
plt.subplots_adjust(hspace=0.5)

for i, col in enumerate(numerical_features, 1):
    plt.subplot(3, 2, i)
    sns.histplot(data=df, x=col, hue='Attrition', kde=True, bins=30, multiple='stack', palette='Pastel2')
    plt.title(f'Distribution of {col} by Attrition', fontsize=14)
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Visualize attrition vs. Monthly Income by Education level
plt.figure(figsize=(14, 8))
sns.violinplot(data=df, x='Education', y='MonthlyIncome', hue='Attrition', split=True, palette='coolwarm')
plt.title('Monthly Income Distribution by Education and Attrition', fontsize=16)
plt.ylabel('Monthly Income ($)')
plt.xlabel('Education Level')
plt.grid(True)
plt.show()

# Visualize a custom example from the PDF: Distance from Home vs. Job Role and Attrition
plt.figure(figsize=(16, 8))
sns.boxplot(data=df, x='JobRole', y='DistanceFromHome', hue='Attrition', palette='Pastel1')
plt.title('Distance from Home by Job Role and Attrition', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Distance From Home (miles)')
plt.xlabel('Job Role')
plt.grid(True)
plt.show()

# 5. Project Summary of Findings
# ------------------------------------------------------------------------------
# Based on the EDA, the following key factors are identified as potential
# drivers of employee attrition:
#
# - Marital Status: Single employees show a significantly higher attrition rate
#   compared to married or divorced employees.
# - Job Role: Laboratory Technicians and Sales Representatives have a higher
#   propensity to leave the company.
# - Business Travel: Employees who travel frequently tend to have a higher
#   attrition rate.
# - Monthly Income: The distribution of Monthly Income for employees who left
#   is notably lower than for those who stayed, especially at the lower income levels.
# - Years at Company: A significant number of employees with fewer years at the
#   company (e.g., 1-2 years) are more likely to leave.
# - Job Satisfaction: Employees with 'Low' or 'Medium' job satisfaction have a
#   higher attrition rate.
# - Distance from Home: Employees with a higher distance from home tend to have
#   a higher attrition rate.
#
# These findings suggest that addressing compensation, job role challenges,
# and work satisfaction could be key strategies for reducing attrition.
