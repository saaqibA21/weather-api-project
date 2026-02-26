import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


df = pd.read_csv("merged_weather_cases.csv")  # Replace with your data

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Features and target
X = df[["temp_c", "humidity", "rain_mm", "wind_kmh"]]   # weather features
y = df["cases"]                                         # next-day cases


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)


model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)  # predicting number of cases
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1,
    verbose=1
)


preds = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)

print("ANN MAE:", mae)
print("ANN MSE:", mse)


plt.figure(figsize=(12,5))
plt.plot(y_test.values, label="Actual")
plt.plot(preds, label="Predicted")
plt.legend()
plt.title("Disease Outbreak Prediction (ANN)")
plt.show()

model.save("ann_outbreak_model.h5")
