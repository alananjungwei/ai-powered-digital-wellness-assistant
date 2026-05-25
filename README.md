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

Several custom features were engineered to improve predictive performance. 

### Screen Sleep Ratio

```py
device_hours_per_day / sleep_hours
```
Measures balance between screen usage and sleep.

### Social Media Ratio

```py
social_media_mins / total_device_time
```
Represents the proportion of device usage spent on social media.

### Notifications Per Hour

```py
notifications_per_day / device_hours_per_day
```
Captures interruption intensity.

### Mental Burden 

```py
(anxiety_score + depression_score + stress_score) / 3
```
Aggregated psychological burden score.

### Well-being Index

```py
(happiness_score + focus_score + sleep_quality_score) / 3
```
Combined well-being measure.

### Digital Physical Balance 

```py
device_hours_per_day / physical_activity_days
```
Balance between digital engagement and physical activity.

### Activity Sleep Interaction
```py
physical_activity_days * sleep_hours
```
Captures interaction between exercise and sleep habits. 

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
## Classification
* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matric

## Regression
* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

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
├── data/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Classification.ipynb
│   ├── Regression.ipynb
│
├── models/
│   ├── high_risk_classifier.pkl
│   ├── digital_dependence_regressor.pkl
│   ├── digital_dependence_scaler.pkl
│   └── model_columns.pkl
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
