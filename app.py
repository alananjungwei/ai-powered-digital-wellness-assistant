import streamlit as st
import joblib
import pandas as pd
from openai import OpenAI

# Load models
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
ada_model = joblib.load(MODELS_DIR / "high_risk_classifier.pkl")
linreg = joblib.load(MODELS_DIR / "digital_dependence_regressor.pkl")
scaler = joblib.load(MODELS_DIR / "digital_dependence_scaler.pkl")
model_columns = joblib.load(MODELS_DIR / "model_columns.pkl")


st.title(
    "🌿 AI Digital Wellness Assistant"
)

st.markdown(
"""
This wellness assistant estimates:

• Digital Dependence Score

• Wellness Risk Level

based on your digital habits,
lifestyle, and wellbeing indicators.
"""
)

#st.success("Models loaded successfully!")

# Level 1

st.header("👤 About You")

user_name = st.text_input(
    "What should I call you?"
)

if user_name:
    st.success(
        f"Hello {user_name}! 👋"
    )

st.caption(
    "Answer a few questions below to receive your personalized digital wellness assessment."
)

gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)


age = st.number_input(
    "Age",
    min_value=13,
    max_value=80,
    value=30
)

region = st.selectbox(
    "Region",
    [
        "Africa",
        "Asia",
        "Europe",
        "Middle East",
        "North America",
        "South America"
    ]
)

education_level = st.selectbox(
    "Education Level",
    [
        "High School",
        "Bachelor",
        "Master",
        "PhD"
    ]
)

income_level = st.selectbox(
    "Income Level",
    [
        "Low",
        "Lower-Mid",
        "Upper-Mid",
        "High"
    ]
)

daily_role = st.selectbox(
    "Primary Daily Role",
    [
        "Student",
        "Full-time Employee",
        "Part-time/Shift",
        "Caregiver/Home",
        "Unemployed_Looking"
    ]
)

st.header("📱 Digital Habits")

device_type = st.selectbox(
    "Primary Device",
    [
        "Android",
        "iPhone",
        "Laptop",
        "Tablet"
    ]
)

device_hours_text = st.select_slider(
    "How much time do you spend on digital devices each day?",
    options=[
        "Very Little",
        "Light Usage",
        "Moderate Usage",
        "Heavy Usage",
        "Almost All Day"
    ],
    value="Moderate Usage"
)

device_hours_map = {
    "Very Little": 1,
    "Light Usage": 4,
    "Moderate Usage": 8,
    "Heavy Usage": 12,
    "Almost All Day": 16
}

device_hours_per_day = device_hours_map[
    device_hours_text
]

phone_unlocks_text = st.select_slider(
    "How often do you find yourself reaching for your phone?",
    options=[
        "Hardly Ever",
        "Once in a While",
        "Fairly Often",
        "Very Often",
        "Almost Constantly"
    ],
    value="Fairly Often"
)

phone_unlocks_map = {
    "Hardly Ever": 0,
    "Once in a While": 75,
    "Fairly Often": 150,
    "Very Often": 250,
    "Almost Constantly": 350
}

phone_unlocks = phone_unlocks_map[
    phone_unlocks_text
]

st.header("🌙 Sleep & Lifestyle")

physical_activity_days = st.slider(
    "Physical Activity Days Per Week",
    0,
    7,
    3
)

sleep_hours = st.slider(
    "Sleep Hours Per Night",
    3.0,
    12.0,
    7.0
)


sleep_quality_text = st.select_slider(
    "How would you rate your sleep quality recently?",
    options=[
        "Very Poor",
        "Poor",
        "Average",
        "Good",
        "Excellent"
    ],
    value="Good"
)

sleep_quality_map = {
    "Very Poor": 1,
    "Poor": 2,
    "Average": 3,
    "Good": 4,
    "Excellent": 5
}

sleep_quality = sleep_quality_map[sleep_quality_text]

st.header("🧠 Mental Wellbeing")

anxiety_text = st.select_slider(
    "How anxious have you felt recently?",
    options=[
        "Very Calm",
        "Slightly Worried",
        "Moderately Anxious",
        "Quite Anxious",
        "Extremely Anxious"
    ],
    value="Moderately Anxious"
)

anxiety_map = {
    "Very Calm": 0,
    "Slightly Worried": 6,
    "Moderately Anxious": 12,
    "Quite Anxious": 18,
    "Extremely Anxious": 25
}

anxiety_score = anxiety_map[anxiety_text]

# Depression Score

depression_text = st.select_slider(
    "How low or down have you been feeling recently?",
    options=[
        "Very Positive",
        "Occasionally Down",
        "Somewhat Low",
        "Frequently Down",
        "Extremely Low"
    ],
    value="Somewhat Low"
)

depression_map = {
    "Very Positive": 0,
    "Occasionally Down": 6,
    "Somewhat Low": 12,
    "Frequently Down": 18,
    "Extremely Low": 25
}

depression_score = depression_map[depression_text]

# Stress level

stress_text = st.select_slider(
    "How stressed do you feel in your daily life?",
    options=[
        "Very Relaxed",
        "Mostly Relaxed",
        "Moderately Stressed",
        "Highly Stressed",
        "Overwhelmed"
    ],
    value="Moderately Stressed"
)

stress_map = {
    "Very Relaxed": 1,
    "Mostly Relaxed": 3,
    "Moderately Stressed": 5,
    "Highly Stressed": 8,
    "Overwhelmed": 10
}

stress_level = stress_map[stress_text]

# Happiness score 

happiness_text = st.select_slider(
    "How happy and satisfied have you felt recently?",
    options=[
        "Very Unhappy",
        "Somewhat Unhappy",
        "Neutral",
        "Happy",
        "Very Happy"
    ],
    value="Neutral"
)

happiness_map = {
    "Very Unhappy": 0,
    "Somewhat Unhappy": 3,
    "Neutral": 5,
    "Happy": 8,
    "Very Happy": 10
}

happiness_score = happiness_map[happiness_text]

# Focus score 

focus_text = st.select_slider(
    "How well have you been able to concentrate recently?",
    options=[
        "Cannot Focus",
        "Easily Distracted",
        "Average Focus",
        "Good Focus",
        "Laser Focused"
    ],
    value="Average Focus"
)

focus_map = {
    "Cannot Focus": 0,
    "Easily Distracted": 25,
    "Average Focus": 50,
    "Good Focus": 75,
    "Laser Focused": 100
}

focus_score = focus_map[focus_text]

# Button 

if st.button("Analyze"):

    # ---------------------
    # Base features
    # ---------------------

    user_df = pd.DataFrame({

        "id":[0],

        "age":[age],

        "device_hours_per_day":[
            device_hours_per_day
        ],

        "phone_unlocks":[
            phone_unlocks
        ],

        "notifications_per_day":[80],

        "social_media_mins":[120],

        "study_mins":[60],

        "physical_activity_days":[
            physical_activity_days
        ],

        "sleep_hours":[sleep_hours],

        "sleep_quality":[sleep_quality],
        "anxiety_score":[anxiety_score],
        "depression_score":[depression_score],
        "stress_level":[stress_level],
        "happiness_score":[happiness_score],
        "focus_score":[focus_score],

    })

    # ---------------------
    # Engineered features
    # ---------------------

    user_df["screen_sleep_ratio"] = (
        user_df["device_hours_per_day"]
        /
        (user_df["sleep_hours"] + 0.1)
    )

    user_df["social_media_ratio"] = (
        user_df["social_media_mins"]
        /
        (
            user_df["device_hours_per_day"] * 60
            + 1
        )
    )

    user_df["notifications_per_hour"] = (
        user_df["notifications_per_day"]
        /
        (
            user_df["device_hours_per_day"]
            + 0.1
        )
    )

    user_df["mental_burden"] = (
        user_df["anxiety_score"]
        +
        user_df["depression_score"]
        +
        user_df["stress_level"]
    ) / 3

    user_df["wellbeing_index"] = (
        user_df["happiness_score"]
        +
        user_df["focus_score"]
        +
        user_df["sleep_quality"]
    ) / 3

    user_df["digital_physical_balance"] = (
        user_df["device_hours_per_day"]
        /
        (
            user_df["physical_activity_days"]
            + 1
        )
    )

    user_df["activity_sleep"] = (
        user_df["physical_activity_days"]
        *
        user_df["sleep_hours"]
    )

    # ---------------------
    # One-hot columns
    # ---------------------

    user_df["gender_Female"] = (
        1 if gender == "Female" else 0
    )

    user_df["gender_Male"] = (
        1 if gender == "Male" else 0
    )

    for col in [
        "region_Africa",
        "region_Asia",
        "region_Europe",
        "region_Middle East",
        "region_North America",
        "region_South America"
    ]:
        user_df[col] = 0

    user_df[f"region_{region}"] = 1

    for col in [
        "income_level_Low",
        "income_level_Lower-Mid",
        "income_level_Upper-Mid",
        "income_level_High"
    ]:
        user_df[col] = 0

    user_df[f"income_level_{income_level}"] = 1

    for col in [
        "education_level_High School",
        "education_level_Bachelor",
        "education_level_Master",
        "education_level_PhD"
    ]:
        user_df[col] = 0

    user_df[f"education_level_{education_level}"] = 1

    for col in [
        "daily_role_Student",
        "daily_role_Full-time Employee",
        "daily_role_Part-time/Shift",
        "daily_role_Caregiver/Home",
        "daily_role_Unemployed_Looking"
    ]:
        user_df[col] = 0

    user_df[f"daily_role_{daily_role}"] = 1

    for col in [
        "device_type_Android",
        "device_type_iPhone",
        "device_type_Laptop",
        "device_type_Tablet"
    ]:
        user_df[col] = 0

    user_df[f"device_type_{device_type}"] = 1

    # ---------------------
    # Match training columns
    # ---------------------

    user_df = user_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    # ---------------------
    # Classification Prediction
    # ---------------------

    prediction = ada_model.predict(
        user_df
    )[0]

    probability = (
        ada_model.predict_proba(user_df)
    )[0,1]


    # ---------------------
    # Regression Prediction
    # ---------------------
    #st.write(user_df.shape)

    user_scaled = scaler.transform(
        user_df
    )

    dependence_score = linreg.predict(
        user_scaled
    )[0]


    # ---------------------
    # Display
    # ---------------------

    st.subheader("📊 Results")

    if prediction == 1:

        st.error(
            f"⚠️ High Risk Detected ({probability:.1%})"
        )

    else:

        st.success(
            f"✅ Low Risk ({1-probability:.1%})"
        )


    if prediction == 1:

        st.info(
            f"""
            {user_name}, this result is only an
            educational wellness estimate and not a
            medical diagnosis.

            Consider prioritizing sleep, regular
            movement, and healthy digital habits.
            """
        )

    else:

        st.info(
            f"""
            Nice work, {user_name}.

            Your current habits appear supportive of
            overall wellbeing. Consistency is key.
            """
        )    

    st.write(
        f"Risk Probability: {probability:.1%}"
    )

    st.metric(
        "Digital Dependence Score",
        f"{dependence_score:.1f}"
    )

    # ---------------------
    # Recommendations
    # ---------------------

    recommendations = []

    if sleep_hours < 7:
        recommendations.append(
            "😴 Try aiming for 7–9 hours of sleep each night to support recovery, mood, and concentration."
        )

    if sleep_quality <= 2:
        recommendations.append(
            "🌙 Improving sleep quality may positively impact your wellbeing and daily energy levels."
        )

    if physical_activity_days < 3:
        recommendations.append(
            "🏃 Consider adding more physical activity throughout the week, even short walks can help."
        )

    if device_hours_per_day >= 12:
        recommendations.append(
            "📱 Your screen time appears quite high. Regular device-free breaks may reduce mental fatigue."
        )

    if phone_unlocks >= 250:
        recommendations.append(
            "📵 Frequent phone checking may contribute to distraction. Consider reducing unnecessary notifications."
        )

    if stress_level >= 8:
        recommendations.append(
            "🧘 Elevated stress levels detected. Relaxation techniques and regular breaks may help."
        )

    if anxiety_score >= 18:
        recommendations.append(
            "💙 You reported high anxiety levels. Reaching out to trusted people and maintaining healthy routines may be beneficial."
        )

    if focus_score <= 25:
        recommendations.append(
            "🎯 A distraction-free environment may help improve focus and productivity."
        )

    if happiness_score <= 3:
        recommendations.append(
            "😊 Spending time outdoors, exercising, or connecting socially may help improve overall wellbeing."
        )

    if recommendations:

        st.subheader(
            "🌱 Personalized Recommendations"
        )

        for rec in recommendations:
            st.write(rec)

    else:

        st.success(
            "Great job! No major lifestyle concerns were detected from the information provided."
        )

    strengths = []

    if sleep_hours >= 7:
        strengths.append("😴 Healthy sleep duration")

    if physical_activity_days >= 4:
        strengths.append("🏃 Good physical activity habits")

    if happiness_score >= 8:
        strengths.append("😊 Positive wellbeing")

    if focus_score >= 75:
        strengths.append("🎯 Strong concentration ability")

    if device_hours_per_day <= 4:
        strengths.append("📱 Balanced screen usage")

    if strengths:

        st.subheader(
            "🌟 Positive Habits"
        )

        for item in strengths:
            st.write(item)