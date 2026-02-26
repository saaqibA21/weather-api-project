import pandas as pd
import numpy as np
import joblib

# Load model + encoders
model = joblib.load("disease_classifier.pkl")
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

print("✅ Disease Outbreak Predictor\n")

# Ask user for manual weather inputs
temp_max = float(input("Enter today's maximum temperature (°C): "))
temp_min = float(input("Enter today's minimum temperature (°C): "))
rain_mm  = float(input("Enter today's rainfall (mm): "))
wind_kmh = float(input("Enter today's wind speed (km/h): "))

# Prepare input for model
X = pd.DataFrame({
    "temp_max": [temp_max],
    "temp_min": [temp_min],
    "rain_mm": [rain_mm],
    "wind_kmh": [wind_kmh]
})

# Scale
X_scaled = scaler.transform(X)

# Predict class
pred = model.predict(X_scaled)[0]
disease = encoder.inverse_transform([pred])[0]

# Predict probabilities
probs = model.predict_proba(X_scaled)[0]

print("\n✅ **Most Likely Disease Outbreak:**", disease)
print("\n✅ **Prediction Probabilities:**\n")
for cls, p in zip(encoder.classes_, probs):
    print(f"{cls:<12}: {p:.3f}")

print("\n✅ Prediction complete!")
