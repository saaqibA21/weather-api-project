import pandas as pd
import numpy as np

# Load weather data
df = pd.read_csv("weather.csv")

np.random.seed(42)

diseases = []

for i, row in df.iterrows():
    temp = row["temp_max"]
    rain = row["rain_mm"]
    wind = row["wind_kmh"]

    # ---- Risk Rules ----
    # Dengue
    dengue_risk = 0
    if 25 <= temp <= 32 and 5 <= rain <= 40:
        dengue_risk = np.random.uniform(0.6, 1.0)

    # Malaria
    malaria_risk = 0
    if rain > 20 and temp >= 24:
        malaria_risk = np.random.uniform(0.6, 1.0)

    # Chikungunya
    chik_risk = 0
    if 26 <= temp <= 34 and rain < 50:
        chik_risk = np.random.uniform(0.5, 0.9)

    # Cholera
    cholera_risk = 0
    if rain > 30:
        cholera_risk = np.random.uniform(0.4, 0.9)

    # Pick highest probability as label
    risks = {
        "dengue": dengue_risk,
        "malaria": malaria_risk,
        "chikungunya": chik_risk,
        "cholera": cholera_risk
    }

    best_disease = max(risks, key=risks.get)

    diseases.append(best_disease)

df["disease"] = diseases

df.to_csv("multi_disease_dataset.csv", index=False)

print("✅ Multi-disease dataset created!")
print(df.head())
