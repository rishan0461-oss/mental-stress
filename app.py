import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
with open("stress_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)

st.set_page_config(page_title="Student Stress Predictor", layout="centered")

st.title("🎓 Student Stress Level Prediction")
st.write("This tool provides early stress awareness based on daily lifestyle inputs.")
st.warning("⚠️ This is not a medical diagnosis tool.")

# User Inputs
sleep_hours = st.slider("🛌 Sleep Hours per Day", 0.0, 10.0, 7.0)
study_hours = st.slider("📚 Study Hours per Day", 0.0, 12.0, 6.0)
screen_time = st.slider("📱 Screen Time per Day", 0.0, 12.0, 5.0)
academic_pressure = st.selectbox("🎯 Academic Pressure Level", [1, 2, 3, 4, 5])

social_interaction = st.selectbox(
    "🧑‍🤝‍🧑 Social Interaction Level",
    ["Low", "Medium", "High"]
)

exercise_frequency = st.selectbox(
    "🏃 Exercise Frequency",
    ["No_Exercise", "Occasional", "Regular"]
)

mood_rating = st.selectbox("😊 Mood Rating", [1, 2, 3, 4, 5])

# Predict Button
if st.button("Predict Stress Level"):
    input_data = pd.DataFrame([{
        "sleep_hours": sleep_hours,
        "study_hours": study_hours,
        "screen_time": screen_time,
        "academic_pressure": academic_pressure,
        "social_interaction": social_interaction,
        "exercise_frequency": exercise_frequency,
        "mood_rating": mood_rating
    }])

    # Encode categorical inputs
    for col in ["social_interaction", "exercise_frequency"]:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Predict
    prediction_encoded = model.predict(input_data)
    prediction = target_encoder.inverse_transform(prediction_encoded)[0]

    # Display result
    if prediction == "Low":
        st.success("🟢 Stress Level: LOW")
        st.write("You're managing well. Keep maintaining healthy habits!")
    elif prediction == "Moderate":
        st.warning("🟡 Stress Level: MODERATE")
        st.write("Consider improving sleep, exercise, and reducing screen time.")
    else:
        st.error("🔴 Stress Level: HIGH")
        st.write("High stress detected. Please consider reaching out to friends, mentors, or counselors.")
