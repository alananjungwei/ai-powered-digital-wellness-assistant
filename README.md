# 🌿AI-Powered Digital Wellness Assistant
An end-to-end Machine Leaning, LLM, and Generative AI application that assesses digital wellnes habits, predicts wellness risk levels, estimates digital dependence, and provides personalized recommendations through an interactive Streamlit user interface.

# 📌Project Overview

The increasing use to digital devices and impact sleep, mental well-being, productivity, and overall lifestyle balance. 

This project aims to help users better understand their digital habits and effects by

* Predicting wellness risk levels
* Estimating digital dependence
* Providing personalized recommendations
* Generating AI-powered wellness feedback (optional OpenAI version)

The application combines traditional Machine Learning, Explainable AI (SHAP), Feature Engineering, and Strealimt deployment into a user-friendly wellness assessment tool. 

# 🎯Objectives
The project focuses on three target variables from the dataset:

## Classification Target
### High Risk Flag

Predict whether a user belongs to a high-risk wellness category based on: 

* Digital behaviour
* Lifestyle habits
* Mental well-being indicators
* Demographic information

## Regression Targets
### Digital Dependence Score
Estimate a user's level of digital dependence using behavioural and well-being indicators.

### Productivity Score
Predict overall productivity levels.

Although multiple models were evaluated, the dataset contained weak predictive signals for productivity score, resulting in limited perspective performance compared to the other two targets. 


# 🧠Machine Learning Workflow 

## 1. Exploratory Data Analysis

Performed exploratory analysis to: 

* Understand feature distributions
* Identify relationships between variables
* Examine correlations
* Detect potential data leakage

Tools:
* Pandas
* Matplotlib
* Seaborn

## 2. Data Pre-processing

#### Handing Categorical Variables
Used one-hot encoding:

```py
pd.get_dummies()
```

for:
* Gender
* Region
* Education Level
* Income Level
* Daily Role
* Device Type

#### Scaling
Applied:
```py
StandardScaler()
```
for models sensitive to feature scale:
* Logsitic Regression
* KNN 
* Linear Regression

#### Train/Test Split

Classification:
```py
train_test_split(stratify = y)
```

Regression:
```py
train_test_split()
```

# Feature Engineering 

Several engineered features were created to improve predictive performance:

|Feature|Description|
|-------|-----------|
|screen_sleep_ratio|Device hours relative to sleep duration|
|social_media_ratio|Social media usage proportion|
|notifications_per_hour|Notification intensity
|mental_burden|Combined anxiety, depression, and stress indicator|
|wellbeing_index|Combined happiness, focus, and sleep quality|
|digital_physical_balance|Digital activity versus physical activity|
|activity_sleep|Interaction between activity and sleep|


# Models Evaluated

## Classification Models
* Logistic Regression
* K-Nearest Neighbours (KNN)
* XGBoost Classifier
* Random Forest Classifier
* AdaBoost Classifier
* Gradient Boosting Classifier

## Regerssion Models 
* Linear Regression
* KNN Regressor
* XGBoost Regressor
* Random Forest Regressor
* AdaBoost Regressor
* Gradient Boosting Regressor

# Hyperparameter Tuning 
Performed optimization using: 
```py
RandomizedSearchCV
```

Parameters tuned included:
* n_estimators
* learning_rate
* max_depth
* min-samples_split
* colsample_bytree
* max_features

depending on the model.

# Evaluation Metrics

|Classification|Regression|
|--------------|----------|
|Accuracy|MAE (Mean Absolute Error)|
|Precision| MSE (Mean Squared Error)|
|Recall|RMSE (Root Mean Squared Error)|
|F1-Score|R² Score|
|ROC-AUS||


# Explainable AI (SHAP)

Model interpretability was implemented using SHAP to identify:
* Most influential risk factors
* Positive and negative contributions
* Feature impact on individual predictions

#### Examples of important features included:
* Stress Level
* Device Usage 
* Sleep Hours
* Focus Score
* Physical Activity

# Large Language Model (LLM) Explanation Layer

Machine learning predictions can be difficult for non-technical users to understand. To improve transparency and usability, OpenAI language models are used to translate model outputs into human-readable explanations. 
### Inputs
The LLM receives: 
* High Risk Probability
* Digital Dependence Score
* Key well-being indicators
* SHAP-identified important features
* User-specific feature values
### Outputs
The LLM generates:
* Plain-language interpretation of results
* Identification of positive habits
* Areas that may deserve attention
* Personalized recommendations

# Generative AI Wellness Coach 
Beyond explaining predictions, the application uses Generative AI to provide personalized wellness guidance tailored to each user's digital habits and well-being profile.

### Inputs
The model considers:
* Risk probability
* Digital dependence score 
* Sleep duration and quality
* Physical activity levels
* Stress and anxiety indicators
* Happiness and focus scores
### Outputs
The Generative AI Wellness Coach produces:

### Wellness Summary
* Friendly and supportive feedback
* Positive observations
* Areas for improvement
* Practical suggestions

### Personalized 7-Day Wellness Plan

The assistant generates a customized action plan including:

* Daily wellbeing activities
* Sleep improvement strategies
* Digital wellbeing recommendations
* Stress management techniques
* Physical activity goals

# Streamlit User Interface

This project includes two deployed user interafaces:

## Version 1: Machine Learning Wellness Assistant
Features:
* User-friendly wellness questionnaire
* Risk prediction
* Digital dependence estimation
* Personalized rule-based recommendations

## Version 2: AI-Powered Wellness Assistant
Additional features:
* OpenAI integration
* AI-generated wellness feedback
* Personalized coaching-style explanations
* Natural langauge wellness recommendations

# Technologies Used 
### Data Science
* Python
* Pandas
* NumPy

### Visualization
* Matplotlib
* Seaborn

### Machine Learning
* Scikit-Learn
* XGBoost

### Explainability 
* SHAP

### Deployment
* Streamlit
* Joblib

### LLM

### Generative AI
* OpenAI API

# Project Structure

```py
AI-Digital-Wellness-Assistant/

│
├── notebooks/
│   ├── notebook_ml.ipynb
│
├── models/
│   ├── high_risk_classifier.pkl
│   ├── digital_dependence_regressor.pkl
│   ├── digital_dependence_scaler.pkl
│   └── model_columns.pkl
│
├── images/
│
├── app.py
│
├── app_openai.py
│
├── requirements.txt
│
└── README.md
```

# Examples 
