import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("multi_disease_dataset.csv")

# Remove non-numeric column
df = df.drop("date", axis=1)

X = df.drop("disease", axis=1)
y = df["disease"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, "xgb_label_encoder.pkl")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "xgb_scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# XGBoost Model
xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(set(y_encoded)),
    eval_metric="mlogloss"
)

xgb.fit(X_train, y_train)

joblib.dump(xgb, "xgboost_model.pkl")

print("XGBoost model trained and saved successfully.")
