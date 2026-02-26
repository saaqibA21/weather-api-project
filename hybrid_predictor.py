# hybrid_predictor.py
import joblib
import numpy as np
import torch
import torch.nn as nn


# -------------------------------------------------------------------
# MUST MATCH TRAINING FEATURES ORDER EXACTLY
# -------------------------------------------------------------------
FEATURE_COLUMNS = [
    "temp_max",
    "temp_min",
    "temp_avg",
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


# -------------------------------------------------------------------
# ANN MODEL (same as training)
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
        x = self.fc3(x)
        return x


# -------------------------------------------------------------------
# LOAD ARTIFACTS
# -------------------------------------------------------------------
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_transformer.pkl")

rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")

num_classes = len(label_encoder.classes_)
pca_input_size = pca.n_components_  # reduced dimension count

ann_model = ANNModel(input_size=pca_input_size, num_classes=num_classes)
ann_model.load_state_dict(torch.load("ann_model.pth", map_location="cpu"))
ann_model.eval()


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def compute_risk_score(ann_probabilities):
    """Risk score from top ANN probability (0–100)."""
    return int(float(np.max(ann_probabilities)) * 100)


def _majority_vote(votes):
    """Returns the class id with majority. If tie, prefers ANN."""
    # votes = [ann, rf, xgb]
    # If tie case: pick ann
    unique = set(votes)
    counts = {v: votes.count(v) for v in unique}
    max_count = max(counts.values())
    top = [k for k, c in counts.items() if c == max_count]
    if len(top) == 1:
        return top[0]
    return votes[0]  # ANN preference


# -------------------------------------------------------------------
# MAIN PREDICT
# -------------------------------------------------------------------
def predict_hybrid(feature_vector):
    """
    feature_vector must be in EXACT order of FEATURE_COLUMNS.
    Example: [temp_max, temp_min, temp_avg, humidity, ... aqi]
    """

    X = np.array(feature_vector, dtype=float).reshape(1, -1)

    if X.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} features, got {X.shape[1]}. "
            f"Check FEATURE_COLUMNS order."
        )

    # 1) Scale
    X_scaled = scaler.transform(X)

    # 2) PCA reduce
    X_pca = pca.transform(X_scaled)

    # 3) ANN probabilities
    with torch.no_grad():
        logits = ann_model(torch.tensor(X_pca, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        ann_pred = int(np.argmax(probs))

    # 4) RF / XGB
    rf_pred = int(rf_model.predict(X_pca)[0])
    xgb_pred = int(xgb_model.predict(X_pca)[0])

    # 5) Majority vote
    votes = [ann_pred, rf_pred, xgb_pred]
    final_class = _majority_vote(votes)
    final_label = label_encoder.inverse_transform([final_class])[0]

    return {
        "final_prediction": final_label,
        "votes": {
            "ann": ann_pred,
            "rf": rf_pred,
            "xgb": xgb_pred
        },
        "ann_probabilities": probs.tolist(),
        "risk_score": compute_risk_score(probs)
    }
