import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("merged_weather_cases.csv")
df["date"] = pd.to_datetime(df["date"])

# FEATURES (X)
X = df[["temp_max", "temp_min", "rain_mm", "wind_kmh"]]

# TARGET (y)
y = df["cases"]

# ---------------------------
# SCALE INPUT FEATURES
# ---------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# TRAIN / TEST SPLIT (time-based)
# ---------------------------
split_idx = int(len(df) * 0.8)

X_train = X_scaled[:split_idx]
X_test  = X_scaled[split_idx:]

y_train = y[:split_idx]
y_test  = y[split_idx:]

# ---------------------------
# BUILD ANN MODEL
# ---------------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(4,)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ---------------------------
# TRAIN MODEL
# ---------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1,
    verbose=1
)

# ---------------------------
# PREDICT ON TEST DATA
# ---------------------------
pred = model.predict(X_test).flatten()

# ---------------------------
# PLOT RESULT
# ---------------------------
plt.figure(figsize=(12,5))
plt.plot(y_test.values, label="Actual Cases", marker="o")
plt.plot(pred, label="Predicted Cases", marker="x")
plt.title("Disease Outbreak Prediction (ANN)")
plt.xlabel("Time")
plt.ylabel("Cases")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------
# SAVE MODEL + SCALER
# ---------------------------
model.save("ann_model.h5")
import joblib
joblib.dump(scaler, "scaler.joblib")

print("✅ ANN training completed")
print("✅ Model saved as ann_model.h5")
print("✅ Scaler saved as scaler.joblib")
