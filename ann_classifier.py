# train_classifiers.py
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
df = pd.read_csv("weather_disease_dataset_1000_days.csv")

FEATURES = [
    "temp_max","temp_min","rain_mm","wind_kmh",
    "humidity","pressure","cloud_cover","dew_point",
    "temp_range","heat_index",
    "rain_3day_avg","rain_7day_sum",
    "month","day_of_year"
]

X = df[FEATURES]
y = df["disease"]

# -------------------------------------------------
# ENCODE LABELS
# -------------------------------------------------
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

# -------------------------------------------------
# SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# -------------------------------------------------
# SCALE FEATURES
# -------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# -------------------------------------------------
# ANN (MLP)
# -------------------------------------------------
ann = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=800,
    random_state=42
)
ann.fit(X_train_s, y_train)

# -------------------------------------------------
# RANDOM FOREST
# -------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42
)
rf.fit(X_train, y_train)

# -------------------------------------------------
# XGBOOST
# -------------------------------------------------
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42
)
xgb.fit(X_train, y_train)

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
print("ANN Report:\n", classification_report(y_test, ann.predict(X_test_s)))
print("RF Report:\n", classification_report(y_test, rf.predict(X_test)))
print("XGB Report:\n", classification_report(y_test, xgb.predict(X_test)))

# -------------------------------------------------
# SAVE MODELS
# -------------------------------------------------
joblib.dump(ann, "ann_model.pkl")
joblib.dump(rf, "rf_model.pkl")
joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ All models trained and saved successfully.")
