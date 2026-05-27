# LLM & Generative AI Architecture

## Overview

This project integrates OpenAI language models to transform machine learning outputs into personalized wellness coaching and actionable recommendations.

The LLM layer does not generate predictions.

Instead, traditional machine learning models first generate:

* Wellness Risk Probability
* Digital Dependence Score

The LLM then:

* Explains predictions
* Highlights strengths and concerns
* Personalizes recommendations
* Generates structured wellness plans

# 🧠 System Architecture

```py
User Inputs
      ↓
Feature Engineering
      ↓
Machine Learning Prediction
      ↓
SHAP-Informed Important Features
      ↓
LLM Prompt Construction
      ↓
AI Wellness Explanation
      ↓
Personalized 7-Day Wellness Plan
```

# 📊 Machine Learning vs LLM Responsibilities

|**Component**|**Responsibility**|
|-------------|------------------|
|Machine Learning Models|Generate predictive outputs|
|SHAP Analysis|Identify influential features|
|LLM Layer|Translate predictions into natural language|
|Generative AI|Create personalized wellness plans|

# 🔍 Prompt Engineering Strategy
Prompt engineering was used to improve:

* Output consistency
* Readability
* Personalization
* Structured formatting
* Response length control

Prompts included:

* Prediction probabilities
* Wellness indicators
* SHAP-informed features
* User wellness goals
* Behavioural patterns

# 📝 AI Wellness Explanation Layer

The AI wellness coach generates:

* Positive habit recognition
* Areas requiring attention
* Practical recommendations
* Encouraging wellness summaries

The assistant was intentionally designed to:

* Avoid medical diagnosis
* Maintain supportive tone
* Provide actionable suggestions
* Use non-technical language

# 📅 Generative AI Wellness Planning 

The LLM also creates:

* Personalized 7-day wellness plans
* Goal-oriented recommendations
* Habit-building suggestions
* Stress and sleep improvement strategies

User goals entered in the application are incorporated into the generated plans.

Example goals:

* Improve sleep quality
* Reduce stress and anxiety
* mprove work-life balance
* Increase focus
* Reduce digital dependence

# Why This Architecture Was the Choice

This architecture combines:

* Predictive power from machine learning
* Transparency from SHAP explainability
* Human-centered interaction through LLMs

The result is a more interpretable and engaging AI wellness experience.

# 📌 Key Design Decisions

## Offline SHAP Analysis

SHAP explanations were generated offline rather than live to simplify deployment and improve performance.

## Controlled LLM Outputs

Prompt constraints were used to:

* Prevent excessive response length
* Improve formatting consistency
* Reduce hallucinations
* Ensure supportive tone

# Human-Centered UX

The application was designed as:

```py
AI-assisted wellness support
```

rather than:

```py
medical diagnosis system
```

This distinction was intentionally maintained throughout the design. 