import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("multi_disease_dataset.csv")

# Drop the date column because RandomForest cannot use string dates directly
df = df.drop("date", axis=1)

# Separate features and target
X = df.drop("disease", axis=1)   # correct column name
y = df["disease"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, "rf_label_encoder.pkl")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "rf_scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

rf.fit(X_train, y_train)

joblib.dump(rf, "random_forest_model.pkl")

print("Random Forest model trained and saved successfully.")
