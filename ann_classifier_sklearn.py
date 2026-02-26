import matplotlib
matplotlib.use("Agg")  # ✅ Use non-GUI backend (no Tkinter required)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("multi_disease_dataset.csv")

# Features
X = df[["temp_max", "temp_min", "rain_mm", "wind_kmh"]]

# Target labels
y = df["disease"]

# Encode text labels → numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Scale weather features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, shuffle=True, random_state=42
)

# ANN model (MLP Neural Network)
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Show report
# Show report
print("\n✅ CLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    y_pred,
    labels=range(len(encoder.classes_)),  # FIXED
    target_names=encoder.classes_
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.title("Confusion Matrix")
plt.show()

# Save model + encoders
joblib.dump(model, "disease_classifier.pkl")
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model training completed and saved successfully!")
