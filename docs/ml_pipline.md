# 🧠 Machine Learning Pipeline

*Project Objective*

The goal of this project was to build an AI-assisted wellness application capable of:

* Predicting wellness risk probability
* Estimating digital dependence levels
* Generating personalized wellness insights

using behavioural, lifestyle, and emotional wellbeing indicators.

# 📊 Dataset & Features 

The dataset included:

* Demographic information
* Digital behaviour indicators
* Sleep and activity patterns
* Emotional wellbeing metrics

## Raw Features 

|**Feature Group**|**Examples**|
|---|---|
|Digital Habits|Device hours, phone unlocks, notifications|
|Lifestyle|Sleep hours, activity levels|
|Mental Wellbeing|Anxiety, stress, focus, happiness|
|Demographics|Age, region, education, income|

# ⚙️ Feature Engineering

Several engineered features were created to improve predictive performance. 

|**Engineered Feature**|**Description**|
|------------------|-----------|
|screen_sleep_ratio	|Relationship between screen time and sleep|
|social_media_ratio	|Social media usage proportion
|notifications_per_hour	|Notification intensity
|mental_burden	|Combined anxiety, depression, and stress indicator
|wellbeing_index	|Combined happiness, focus, and sleep quality score
|digital_physical_balance|	Relationship between screen time and activity
|activity_sleep	|Interaction between activity and sleep

# 🔄 Data Preprocessing 

The preprocessing pipeline included:

* Missing value handling
* Feature engineering
* One-hot encoding for categorical variables
* Feature scaling for regression models
* Column alignment for deployment consistency

Categorical variables such as region, education, income level, and device type were transformed using one-hot encoding.

# 🔧 Hyperparameter Tuning

Hyperparameter optimization was performed using:

```py
RandomizedSearchCV
```
Parameters tuned included:

* n_estimators
* learning_rate
* max_depth
* min_samples_split
* colsample_bytree
* max_features

# Evaluation Metrics

## Classification Metrics

|Metric|	Purpose|
|---|---|
|Accuracy	|Overall prediction correctness|
|Precision|Reliability of positive predictions|
|Recall	|Ability to identify high-risk cases|
|F1-Score	|Balance between precision and recall|
|ROC-AUC	|Overall classification quality|

## Regression Metrics

|Metric	|Purpose|
|---|---|
|MAE	|Average prediction error|
|MSE|	Squared prediction error|
|RMSE	|Error magnitude|
|R² Score	|Explained variance|


# 🤖 Models Evaluated 

## Classification Models
* Logistic Regression
* K-Nearest Neighbours (KNN)
* Random Forest Classifier
* XGBoost Classifier
* AdaBoost Classifier
* Gradient Boosting Classifier

## Regression Models
* Linear Regression
* KNN Regressor
* Random Forest Regressor
* XGBoost Regressor
* AdaBoost Regressor
* Gradient Boosting Regressor

# Model Performance



## AdaBoost Classifier (For risk prediction, Classification)

| Metric | Score |
|----------|----------|
| Accuracy | 0.883 |
| Precision | 0.82 |
| Recall | 0.54 |
| F1 Score | 0.65 |
| ROC-AUC | 0.782 |

## Linear Regression (for digital dependence, Regression)

| Metric | Score |
|----------|----------|
| MAE | 0.708 |
| RMSE | 1.70 |
| R² | 0.986 | 

# Final Model Selection 

|Task	|Final Model|
|---|---|
|Wellness Risk Prediction	|AdaBoost Classifier|
|Digital Dependence Score|	Linear Regression|


The final models were selected based on performance, interpretability, and deployment simplicity.




# 📌 Key Findings

* Behavioural and wellbeing features contributed more strongly than demographic variables.
* Sleep, stress, and device usage were among the strongest predictors.
* Feature engineering significantly improved predictive quality.
* Simpler models provided better deployment efficiency and interpretability.




