import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np

# Load models
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
ada_model = joblib.load(MODELS_DIR / "high_risk_classifier.pkl")
linreg = joblib.load(MODELS_DIR / "digital_dependence_regressor.pkl")
scaler = joblib.load(MODELS_DIR / "digital_dependence_scaler.pkl")
model_columns = joblib.load(MODELS_DIR / "model_columns.pkl")


st.title("🌿 AI Digital Wellness Assistant")

st.markdown(
"""
This wellness assistant estimates:

• Digital Dependence Score

• Wellness Risk Level

based on your digital habits,
lifestyle, and wellbeing indicators.
"""
)

with st.sidebar:

    st.header("🤖 AI Settings")

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password"
    )

#st.success("Models loaded successfully!")

# Level 1
st.header("👤 About You")
st.caption("Step 1 of 4: About You")

user_name = st.text_input("What should I call you?")

if user_name:
    st.success(
        f"Hello {user_name}! 👋"
    )

st.caption("Answer a few questions below to receive your personalized digital wellness assessment.")

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
st.caption("Step 2 of 4: Digital Habits")

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
st.caption("Step 3 of 4: Sleep & Lifestyle")

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
st.caption("Step 4 of 4: Mental Wellbeing")

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

    # Engineered features

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

    # One-hot columns

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

    # match training columns 

    user_df = user_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    # classification prediction 

    prediction = ada_model.predict(
        user_df
    )[0]

    probability = (
        ada_model.predict_proba(user_df)
    )[0,1]


    # regression prediction
    #st.write(user_df.shape)

    user_scaled = scaler.transform(
        user_df
    )

    dependence_score = linreg.predict(
        user_scaled
    )[0]

    # wellness radar chart 

    sleep_score = sleep_quality * 20

    activity_score = (
        physical_activity_days / 7
    ) * 100

    stress_score = (
        (10 - stress_level) / 10
    ) * 100

    focus_chart = focus_score

    happiness_chart = happiness_score * 10
        

    categories = [
        "Sleep",
        "Activity",
        "Low Stress",
        "Focus",
        "Happiness"
    ]

    values = [
        sleep_score,
        activity_score,
        stress_score,
        focus_chart,
        happiness_chart
    ]

    values += values[:1]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(categories),
        endpoint=False
    ).tolist()

    angles += angles[:1]

    fig = plt.figure(figsize=(6,6))

    ax = plt.subplot(
        111,
        polar=True
    )

    ax.plot(
        angles,
        values,
        linewidth=2
    )

    ax.fill(
        angles,
        values,
        alpha=0.25
    )

    ax.set_xticks(
        angles[:-1]
    )

    ax.set_xticklabels(
        categories
    )

    ax.set_ylim(
        0,
        100
    )

    # AI wellness summary
    ai_feedback = None
    ai_explanation = None
    wellness_plan = None
        
    if openai_api_key:

        try:

            client = OpenAI(
                api_key=openai_api_key
            )

            prompt = f"""
            User Name: {user_name}

            High Risk Probability:
            {probability:.1%}

            Digital Dependence Score:
            {dependence_score:.1f}

            Age:
            {age}

            Device Hours:
            {device_hours_per_day}

            Physical Activity Days:
            {physical_activity_days}

            Sleep Hours:
            {sleep_hours}

            Sleep Quality:
            {sleep_quality}

            Anxiety Score:
            {anxiety_score}

            Depression Score:
            {depression_score}

            Stress Level:
            {stress_level}

            Happiness Score:
            {happiness_score}

            Focus Score:
            {focus_score}

            Format the response EXACTLY like this:

            Strengths:
            - bullet point
            - bullet point

            Areas to Improve:
            - bullet point
            - bullet point

            Recommendation:
            - short practical recommendation

            Maximum 120 words.
            Use short bullet points.
            """

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role":"system",
                        "content":
                        """
                        You are a supportive
                        digital wellness coach.

                        Never diagnose.

                        Be encouraging,
                        constructive and practical.
                        """
                    },
                    {
                        "role":"user",
                        "content":prompt
                    }
                ]
            )

            ai_feedback = (
                response
                .choices[0]
                .message
                .content
            )

            # SHAP based factors 

            high_risk_shap_features = [
                "Stress Level",
                "Device Hours Per Day",
                "Sleep Hours",
                "Mental Burden",
                "Screen-to-Sleep Ratio"
            ]

            dependence_shap_features = [
                "Device Hours Per Day",
                "Phone Unlock Frequency",
                "Notifications Per Day",
                "Age",
                "Activity-Sleep Balance"
            ]

            explanation_prompt = f"""
            User:
            {user_name}

            Risk Probability:
            {probability:.1%}

            Digital Dependence Score:
            {dependence_score:.1f}

            Key wellness indicators:

            • Stress Level: {stress_level}/10
            • Sleep Duration: {sleep_hours} hours/night
            • Device Usage: {device_hours_per_day} hours/day
            • Phone Unlocks: {phone_unlocks}/day
            • Physical Activity Days: {physical_activity_days}/week
            • Mental Burden Score: {user_df['mental_burden'].iloc[0]:.1f}
            • Screen-to-Sleep Ratio: {user_df['screen_sleep_ratio'].iloc[0]:.2f}
            • Activity-Sleep Balance: {user_df['activity_sleep'].iloc[0]:.1f}

            These indicators were among the strongest factors
            identified during model interpretation.

            Format exactly as:

            Summary:
            - one short paragraph

            Positive Habits:
            - bullet point
            - bullet point

            Areas for Attention:
            - bullet point
            - bullet point

            Next Step:
            - one practical action

            Maximum 120 words.
            Use bullet points where appropriate.
            """

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                max_tokens= 350,
                messages=[
                    {
                        "role":"system",
                        "content":
                        "You explain wellness predictions in friendly language."
                    },
                    {
                        "role":"user",
                        "content": explanation_prompt
                    }
                ]
            )

            ai_explanation = (
                response
                .choices[0]
                .message
                .content
            )

            plan_prompt = f"""
            Create a personalized 7-day wellness plan.

            User:
            {user_name}

            High Risk Probability:
            {probability:.1%}

            Digital Dependence Score:
            {dependence_score:.1f}

            Sleep Hours:
            {sleep_hours}

            Physical Activity Days:
            {physical_activity_days}

            Stress Level:
            {stress_level}

            Anxiety Score:
            {anxiety_score}

            Happiness Score:
            {happiness_score}

            Focus Score:
            {focus_score}

            Requirements:

            - Create a 7-day plan

            Day 1:
            - one action

            Day 2:
            - one action

            Day 3:
            - one action

            Day 4:
            - one action

            Day 5:
            - one action

            Day 6:
            - one action

            Day 7:
            - one action

            Then provide:
            - 3 additional tips

            Maximum 250 words.

            """

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                max_tokens=300,
                messages=[
                    {
                        "role":"system",
                        "content":
                        """
                        You are a digital wellness coach.

                        Create realistic wellness plans.

                        Never diagnose.

                        Focus on practical habits.
                        """
                    },
                    {
                        "role":"user",
                        "content": plan_prompt
                    }
                ]
            )

            wellness_plan = (
                response
                .choices[0]
                .message
                .content
            )

        except Exception as e:

            st.error(
                f"OpenAI Error: {e}"
            )


    # Display

    st.subheader("📊 Results")

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

    # Wellness Dashboard

    st.subheader("📊 Wellness Dashboard")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🛡️ Wellness Risk")

        st.progress(float(probability))

        st.write(
            f"Risk Probability: {probability:.1%}"
        )

        if probability < 0.20:
            st.success("🟢 Low Risk")
        elif probability < 0.50:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")

    dependence_normalized = min(max(dependence_score / 100, 0), 1)

    with col2:

        st.markdown("### 📱 Digital Dependence")
        st.progress(dependence_normalized)
        st.write(f"Dependence Score: {dependence_score:.1f}")

        if dependence_score < 40:
            st.success("🟢 Low Digital Dependence")
        elif dependence_score < 70:
            st.warning("🟡 Moderate Digital Dependence")
        else:
            st.error("🔴 High Digital Dependence")

    st.subheader(
    "📈 Wellness Profile"
    )

    strengths = []

    if sleep_hours >= 7:
        strengths.append("😴 Healthy sleep duration")

    if physical_activity_days >= 4:
        strengths.append("🏃 Regular physical activity")

    if happiness_score >= 8:
        strengths.append("😊 Positive wellbeing")

    if focus_score >= 75:
        strengths.append("🎯 Strong focus habits")  

    opportunities = []

    if device_hours_per_day >= 12:
        opportunities.append("📱 High daily screen time")

    if phone_unlocks >= 250:
        opportunities.append("🔔 Frequent phone checking")

    if stress_level >= 8:
        opportunities.append("⚠ Elevated stress levels")

    if anxiety_score >= 18:
        opportunities.append("🧠 Higher anxiety levels")

    st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("💪 Your Strengths")

        if strengths:
            for item in strengths:
                st.write(item)

        else:
            st.write("No major strengths identified yet.")

    with col2:

        st.subheader("🌱 Opportunities")

        if opportunities:
            for item in opportunities:
                st.write(item)

        else:
            st.write("No major concerns identified.")
    

    if ai_feedback:

        st.subheader("🤖 AI Wellness Coach")
        st.write(ai_feedback)

    if ai_explanation:


        with st.expander(
            "🔍 Read Full Explanation"
        ):
            st.markdown(ai_explanation)

    if wellness_plan:

        st.subheader("📅 Your Personalized 7-Day Wellness Plan")
        with st.expander(
            "📅 Your Personalized 7-Day Wellness Plan",
            expanded=True
        ):
            st.markdown(wellness_plan)
    
    # SHAP Insights 

    st.subheader(
    "🔍 What Influenced Your Assessment?"
    )

    st.write(
        f"""
    📱 Device Usage:
    {device_hours_per_day} hours/day

    😴 Sleep Duration:
    {sleep_hours} hours/night

    ⚠ Stress Level:
    {stress_level}/10

    🔓 Phone Unlocks:
    {phone_unlocks}/day

    🏃 Physical Activity:
    {physical_activity_days} days/week
    """
    )

    if stress_level >= 8:
        st.warning(
            "Stress appears to be one of the strongest contributors to your wellness risk."
        )

    if sleep_hours < 6:
        st.warning(
            "Your sleep duration may be affecting your overall wellbeing."
        )

    if device_hours_per_day >= 12:
        st.warning(
            "High daily device usage may be contributing to digital dependence."
        )

    st.caption(
        """
    These indicators were among the most influential
    factors identified through SHAP analysis of the
    machine learning models.
    """
    )

    st.caption(
        "These insights were identified through SHAP model interpretation."
    )

    # Download Report
    report_text = f"""
    DIGITAL WELLNESS REPORT

    Name:
    {user_name}

    Risk Probability:
    {probability:.1%}

    Digital Dependence Score:
    {dependence_score:.1f}

    --------------------------------

    STRENGTHS

    {chr(10).join(strengths)}

    --------------------------------

    OPPORTUNITIES

    {chr(10).join(opportunities)}

    --------------------------------

    AI WELLNESS COACH

    {ai_feedback if ai_feedback else "Not generated"}

    --------------------------------

    AI EXPLANATION

    {ai_explanation if ai_explanation else "Not generated"}

    --------------------------------

    7-DAY WELLNESS PLAN

    {wellness_plan if wellness_plan else "Not generated"}
    """

    st.subheader("📥 Export Results")
    st.download_button(
        label="📄 Download Wellness Report",
        data=report_text,
        file_name="digital_wellness_report.txt",
        mime="text/plain"
    )