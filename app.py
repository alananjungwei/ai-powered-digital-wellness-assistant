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
    # Enter own API Key
    st.header("🤖 AI Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password")


# Level 1
st.header("👤 About You")
st.caption("Step 1 of 4: About You")
user_name = st.text_input("What should I call you?")

if user_name:
    st.success(f"Hello {user_name}! 👋")

st.caption("Tell me something about yourself!")

gender = st.radio(
    "Gender",
    [
        "👨 Male",
        "👩 Female",
        "🌈 Non-binary / Other",
        "🙈 Prefer not to say"
    ],
    horizontal=True
)


age = st.number_input(
    "Age",
    min_value=13,
    max_value=80,
    value=30
)

region = st.radio(
    "Where are you based?",
    [
        "🦁 Africa",
        "🐉 Asia",
        "🏰 Europe",
        "🕌 Middle East",
        "🗽 North America",
        "🦜 South America",
        "🦘 Oceania"
    ]
)

education_level = st.radio(
    "Highest education level completed?",
    [
        "🎓 High School",
        "📘 Bachelor",
        "📗 Master",
        "🔬 PhD"
    ],
    horizontal=True
)

income_level = st.radio(
    "How would you describe your current financial situation?",
    [
        "💸 Limited",
        "🙂 Comfortable",
        "💼 Stable",
        "✨ Very Comfortable"
    ],
    horizontal=True
)

daily_role = st.radio(
    "What best describes your daily routine?",
    [
        "🎓 Student",
        "💼 Full-time Employee",
        "🕒 Shift/Part-time",
        "🏡 Caregiver/Home",
        "🔍 Job Seeking"
    ]
)

st.header("📱 Digital Habits")
st.caption("Step 2 of 4: Digital Habits")

device_type = st.radio(
    "What is your primary device?",
    options=[
        "🤖 Android",
        "🍎 iPhone",
        "💻 Laptop",
        "📱 Tablet"
    ],
    horizontal = True
)

device_hours_text = st.radio(
    "How much time do you spend on digital devices each day?",
    options=[
        "🟢 Very Little",
        "🟡 Light Usage",
        "🟠 Moderate Usage",
        "🔴 Heavy Usage",
        "🚨 Almost All Day"
    ],
    horizontal = True
)

device_hours_map = {
    "🟢 Very Little": 1,
    "🟡 Light Usage": 4,
    "🟠 Moderate Usage": 8,
    "🔴 Heavy Usage": 12,
    "🚨 Almost All Day": 16
}

device_hours_per_day = (
    device_hours_map[device_hours_text]
)

phone_unlocks_text = st.radio(
    "How often do you find yourself reaching for your phone?",
    options=[
        "🟢 Hardly Ever",
        "🟡 Once in a While",
        "🟠 Fairly Often",
        "🔴 Very Often",
        "🚨 Almost Constantly"
    ],
    horizontal=True
)

phone_unlocks_map = {
    "🟢 Hardly Ever": 0,
    "🟡 Once in a While": 75,
    "🟠 Fairly Often": 150,
    "🔴 Very Often": 250,
    "🚨 Almost Constantly": 350
}

phone_unlocks = (
    phone_unlocks_map[phone_unlocks_text]
)

st.header("🌙 Sleep & Lifestyle")
st.caption("Step 3 of 4: Sleep & Lifestyle")
st.markdown(
    """
Your daily habits can strongly influence
focus, stress, recovery, and overall wellbeing.
"""
)


physical_activity_text = st.radio(
    "How active have you been recently?",
    [
        "🛋️ Rarely Active",
        "🚶 Lightly Active",
        "🏃 Moderately Active",
        "💪 Very Active"
    ],
    horizontal=True
)

physical_activity_map = {
    "🛋️ Rarely Active": 1,
    "🚶 Lightly Active": 3,
    "🏃 Moderately Active": 5,
    "💪 Very Active": 7
}

physical_activity_days = (
    physical_activity_map[
        physical_activity_text
    ]
)

sleep_text = st.radio(
    "How much sleep do you usually get?",
    [
        "😴 Very Little Sleep",
        "🛌 Slightly Sleep Deprived",
        "🙂 Adequate Sleep",
        "✨ Well Rested"
    ],
    horizontal=True
)

sleep_map = {
    "😴 Very Little Sleep": 4,
    "🛌 Slightly Sleep Deprived": 6,
    "🙂 Adequate Sleep": 7.5,
    "✨ Well Rested": 9
}

sleep_hours = sleep_map[sleep_text]


sleep_quality_text = st.radio(
    "How would you rate your sleep quality recently?",
    [
        "😴 Very Poor",
        "😕 Poor",
        "😐 Average",
        "🙂 Good",
        "✨ Excellent"
    ],
    horizontal=True
)

sleep_quality_map = {
    "😴 Very Poor": 1,
    "😕 Poor": 2,
    "😐 Average": 3,
    "🙂 Good": 4,
    "✨ Excellent": 5
}

sleep_quality = sleep_quality_map[sleep_quality_text]

st.header("🧠 Mental Wellbeing")

st.caption("Step 4 of 4: Mental Wellbeing")

st.markdown(
    """
These questions help the assistant understand
how you've been feeling emotionally and mentally lately.

Remember, there are no right or wrong answers, just choose
what feels most accurate for you.

This is a safe space. You can be honest 🙂. 
"""
)


anxiety_text = st.radio(
    "How have your worries or anxious thoughts felt recently?",
    [
        "😌 Calm & Relaxed",
        "🙂 Occasionally Worried",
        "😐 Somewhat Anxious",
        "😟 Frequently Anxious",
        "😣 Constantly Overwhelmed"
    ],
    horizontal=True
)

anxiety_map = {
    "😌 Calm & Relaxed": 0,
    "🙂 Occasionally Worried": 6,
    "😐 Somewhat Anxious": 12,
    "😟 Frequently Anxious": 18,
    "😣 Constantly Overwhelmed": 25
}

anxiety_score = anxiety_map[
    anxiety_text
]

# Depression Score

depression_text = st.radio(
    "How has your mood been recently?",
    [
        "✨ Positive & Motivated",
        "🙂 Mostly Okay",
        "😐 Emotionally Tired",
        "😔 Frequently Low",
        "😞 Very Drained"
    ],
    horizontal=True
)

depression_map = {
    "✨ Positive & Motivated": 0,
    "🙂 Mostly Okay": 6,
    "😐 Emotionally Tired": 12,
    "😔 Frequently Low": 18,
    "😞 Very Drained": 25
}

depression_score = depression_map[
    depression_text
]

# Stress level

stress_text = st.radio(
    "How demanding or stressful has daily life felt lately?",
    [
        "🌿 Peaceful",
        "🙂 Manageable",
        "😐 Busy but Coping",
        "😟 Quite Stressful",
        "🔥 Completely Overloaded"
    ],
    horizontal=True
)

stress_map = {
    "🌿 Peaceful": 1,
    "🙂 Manageable": 3,
    "😐 Busy but Coping": 5,
    "😟 Quite Stressful": 8,
    "🔥 Completely Overloaded": 10
}

stress_level = stress_map[
    stress_text
]

# Happiness score 

happiness_text = st.radio(
    "How satisfied and emotionally balanced have you felt recently?",
    [
        "😞 Very Unhappy",
        "🙁 Slightly Unhappy",
        "😐 Neutral",
        "🙂 Generally Happy",
        "✨ Very Happy"
    ],
    horizontal=True
)

happiness_map = {
    "😞 Very Unhappy": 0,
    "🙁 Slightly Unhappy": 3,
    "😐 Neutral": 5,
    "🙂 Generally Happy": 8,
    "✨ Very Happy": 10
}

happiness_score = happiness_map[
    happiness_text
]

# Focus score 

focus_text = st.radio(
    "How focused and mentally clear have you felt recently?",
    [
        "😵 Unable to Focus",
        "😵‍💫 Easily Distracted",
        "😐 Average Focus",
        "🙂 Mostly Focused",
        "🎯 Highly Focused"
    ],
    horizontal=True
)

focus_map = {
    "😵 Unable to Focus": 0,
    "😵‍💫 Easily Distracted": 25,
    "😐 Average Focus": 50,
    "🙂 Mostly Focused": 75,
    "🎯 Highly Focused": 100
}

focus_score = focus_map[
    focus_text
]

# Personal goal 
st.header("🎯 Personal Goal")

user_goal = st.text_area(
    "What would you most like to improve?",
    placeholder=
    """
Examples:
• I want to sleep better
• I want to spend less time on my phone
• I want to worry less
• I want to be more productive
• I want to exercise more
• I want to improve my focus
"""
)

# Button 
if st.button("Analyze"):

    # base features

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

            Personal Goal:
            {user_goal}

            Prioritize the user's goal when giving advice.

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

            Personal Goal:
            {user_goal}

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

            Consider the user's personal goal when explaining
            the results and recommendations.

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


            Personal Goal:
            {user_goal}

            The plan should prioritize helping the user
            achieve this goal whenever appropriate.

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

            Before Day 1, include:

            Goal Focus:
            (one sentence explaining how this plan supports
            the user's stated goal)

            Then create the 7-day plan.

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

    Personal Goal:
    {user_goal}

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