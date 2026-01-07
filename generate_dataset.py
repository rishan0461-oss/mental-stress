import pandas as pd
import random

data = []

for _ in range(300):
    sleep = round(random.uniform(4, 9), 1)
    study = round(random.uniform(1, 10), 1)
    screen = round(random.uniform(1, 10), 1)
    pressure = random.randint(1, 5)
    mood = random.randint(1, 5)
    social = random.choice(["Low", "Medium", "High"])
    exercise = random.choice(["No_Exercise", "Occasional", "Regular"])

    # Stress logic
    if sleep < 5 or (pressure >= 4 and mood <= 2):
        stress = "High"
    elif sleep >= 7 and mood >= 4 and exercise == "Regular":
        stress = "Low"
    else:
        stress = "Moderate"

    data.append([
        sleep, study, screen, pressure, social, exercise, mood, stress
    ])

df = pd.DataFrame(data, columns=[
    "sleep_hours",
    "study_hours",
    "screen_time",
    "academic_pressure",
    "social_interaction",
    "exercise_frequency",
    "mood_rating",
    "stress_level"
])

df.to_csv("student_stress_data.csv", index=False)
print("Dataset created successfully!")
