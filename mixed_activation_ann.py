import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ---------------------
# Load data
# ---------------------
df = pd.read_csv("multi_disease_dataset.csv")

X = df[["temp_max", "temp_min", "rain_mm", "wind_kmh"]]
y = df["disease"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# ---------------------
# MODEL WITH MULTIPLE ACTIVATION FUNCTIONS
# ---------------------
class MixedActivationANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

        # Activations
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))       # Layer 1 → ReLU
        x = self.tanh(self.fc2(x))       # Layer 2 → Tanh
        x = self.fc3(x)
        return self.softmax(x)           # Output → Softmax

model = MixedActivationANN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------------
# TRAIN
# ---------------------
for epoch in range(50):
    for batch_X, batch_y in train_loader:
        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print("Epoch", epoch, "Loss:", float(loss))

# ---------------------
# SAVE
# ---------------------
torch.save(model.state_dict(), "mixed_activation_model.pth")
import joblib
joblib.dump(scaler, "pytorch_scaler.pkl")
joblib.dump(encoder, "pytorch_encoder.pkl")

print("✅ Model trained with multiple activation functions!")
