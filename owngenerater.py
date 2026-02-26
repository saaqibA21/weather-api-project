import pandas as pd
import numpy as np

weather = pd.read_csv("weather.csv")

# create synthetic cases based on temperature & rain
np.random.seed(42)
base = 10 + (30 - weather["temp_max"]) * 0.3 + weather["rain_mm"] * 0.2
cases = base + np.random.normal(0, 3, len(base))

# ✅ FIXED HERE
cases = cases.clip(lower=1).astype(int)

df = pd.DataFrame({
    "date": weather["date"],
    "cases": cases
})

df.to_csv("cases.csv", index=False)
print("✅ Synthetic cases.csv created.")
