# 🔍 Model Explainability with SHAP

**Why Explainability Matters**

Machine learning predictions can sometimes feel like black boxes. 

To improve transparency and interpretability, SHAP (SHapley Additive exPlanations) analysis was used to identify which features most strongly influenced predicitons.

This allowed the application to:

* Highlight important behavioural patterns
* Improve user trust
* Support human-friendly explanations
* Connect predictions to actionable wellness insights

# 🧠 Explainability Workflow

```py
Trained ML Models
        ↓
Offline SHAP Analysis
        ↓
Global Feature Importance
        ↓
Important Features Identified
        ↓
Integrated into AI Explanations
```

# 📊 High Risk Prediction Explainability

Most influential features

* Stress Level
* Device Hours Per Day
* Sleep Hours
* Mental Burden
* Screen-to-Sleep Ratio

# 📱Digital Dependence Explainability 

Most influential features

* Device Hours Per Day
* Phone Unlock Frequency
* Notifications Per Day
* Activity-Sleep Balance
* Age

# Key Insights

The SHAP analysis revealed that: 
* Behavioural patterns contributed more strongly than demographic variables.
* Sleep and stress indicators were highly influential wellness predictors.
* Device usage patterns strongly influenced digital dependence.
* Engineered features improved interpretability and model performance.

# Why Offline SHAP Was Used 

Instead of generating SHAP values live inside the application, SHAP analysis was performed offline during development.

This approach:

* Reduced application complexity
* Improved Streamlit performance
* Simplified deployment
* Preserved interpretability insights

The identified important features were later integrated into the LLM explanation prompts.

# Explainability + LLM Integration
SHAP analysis was not directly shown to users.

Instead:

* PSHAP identified influential features
* PImportant factors were injected into prompts
* PThe LLM translated technical insights into user-friendly explanations

This created a more human-centered and understandable AI experience.






















