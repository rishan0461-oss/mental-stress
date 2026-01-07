import pickle
import pandas as pd

# Load model and encoders
with open("stress_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)

print("Model and encoders loaded successfully!")

# Sample input (simulate user input)
input_data = {
    "sleep_hours": 4.5,
    "study_hours": 8,
    "screen_time": 7,
    "academic_pressure": 5,
    "social_interaction": "Low",
    "exercise_frequency": "No_Exercise",
    "mood_rating": 2
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical inputs
for column in ["social_interaction", "exercise_frequency"]:
    input_df[column] = label_encoders[column].transform(input_df[column])

# Make prediction
prediction_encoded = model.predict(input_df)
prediction = target_encoder.inverse_transform(prediction_encoded)

print("Predicted Stress Level:", prediction[0])
