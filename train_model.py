# train_model.py
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------------------------------------------------
# FEATURES USED BY MODEL (15 FEATURES TOTAL)
# NOTE: temp_avg IS DERIVED, NOT EXPECTED IN CSV
# -------------------------------------------------------------------
FEATURE_COLUMNS = [
    "temp_max",
    "temp_min",
    "temp_avg",          # derived
    "rain_mm",
    "wind_kmh",
    "humidity",
    "pressure",
    "cloud_cover",
    "dew_point",
    "temp_range",
    "heat_index",
    "rain_3day_avg",
    "rain_7day_sum",
    "month",
    "day_of_year"
]

RAW_COLUMNS_REQUIRED = [
    "temp_max",
    "temp_min",
    "rain_mm",
    "wind_kmh",
    "humidity",
    "pressure",
    "cloud_cover",
    "dew_point",
    "temp_range",
    "heat_index",
    "rain_3day_avg",
    "rain_7day_sum",
    "month",
    "day_of_year"
]

TARGET_COLUMN = "disease"


# -------------------------------------------------------------------
# ANN MODEL
# -------------------------------------------------------------------
class ANNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


# -------------------------------------------------------------------
# MAIN TRAINING PIPELINE
# -------------------------------------------------------------------
def main():

    if not os.path.exists("weather_disease_dataset.csv"):
        raise FileNotFoundError("weather_disease_dataset.csv not found")

    df = pd.read_csv("weather_disease_dataset.csv")

    # ---------------------------------------------------------------
    # Validate RAW columns (NOT derived ones)
    # ---------------------------------------------------------------
    missing = [c for c in RAW_COLUMNS_REQUIRED + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # ---------------------------------------------------------------
    # DERIVED FEATURES
    # ---------------------------------------------------------------
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2

    # ---------------------------------------------------------------
    # Feature matrix & target
    # ---------------------------------------------------------------
    X = df[FEATURE_COLUMNS].astype(float).values
    y_raw = df[TARGET_COLUMN].astype(str).values

    # ---------------------------------------------------------------
    # Encode labels
    # ---------------------------------------------------------------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    joblib.dump(label_encoder, "label_encoder.pkl")

    print("Disease classes:", list(label_encoder.classes_))

    # ---------------------------------------------------------------
    # Scaling
    # ---------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")

    # ---------------------------------------------------------------
    # PCA (95% variance)
    # ---------------------------------------------------------------
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, "pca_transformer.pkl")

    print(f"Original features : {X.shape[1]}")
    print(f"PCA features      : {X_pca.shape[1]}")

    # ---------------------------------------------------------------
    # Train/Test Split
    # ---------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ===================== ANN =====================
    ann = ANNModel(
        input_size=X_pca.shape[1],
        num_classes=len(label_encoder.classes_)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ann.parameters(), lr=0.001)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(60):
        ann.train()
        optimizer.zero_grad()
        loss = criterion(ann(Xtr), ytr)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"ANN Epoch {epoch:02d} | Loss: {loss.item():.4f}")

    torch.save(ann.state_dict(), "ann_model.pth")

    with torch.no_grad():
        ann_pred = torch.argmax(
            ann(torch.tensor(X_test, dtype=torch.float32)),
            dim=1
        ).numpy()

    # ===================== RF =====================
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, "random_forest_model.pkl")
    rf_pred = rf.predict(X_test)

    # ===================== XGB =====================
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss"
    )
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, "xgboost_model.pkl")
    xgb_pred = xgb.predict(X_test)

    # ===================== HYBRID =====================
    hybrid_pred = [
        max([a, r, x], key=[a, r, x].count)
        for a, r, x in zip(ann_pred, rf_pred, xgb_pred)
    ]

    print("\nAccuracy Scores")
    print("ANN :", accuracy_score(y_test, ann_pred))
    print("RF  :", accuracy_score(y_test, rf_pred))
    print("XGB :", accuracy_score(y_test, xgb_pred))
    print("HYB :", accuracy_score(y_test, hybrid_pred))

    print("\nHybrid Classification Report")
    print(classification_report(
        y_test,
        hybrid_pred,
        target_names=label_encoder.classes_
    ))


if __name__ == "__main__":
    main()
