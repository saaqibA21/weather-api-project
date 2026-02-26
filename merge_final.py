import pandas as pd

# Load both datasets
weather = pd.read_csv("weather.csv")
cases = pd.read_csv("cases.csv")

# Convert to datetime
weather["date"] = pd.to_datetime(weather["date"])
cases["date"] = pd.to_datetime(cases["date"])

# Sort by date (safe)
weather = weather.sort_values("date")
cases = cases.sort_values("date")

# Merge on date
merged = pd.merge(weather, cases, on="date", how="inner")

# Save
merged.to_csv("merged_weather_cases.csv", index=False)

print("✅ merged_weather_cases.csv created successfully!")
print(merged.head())
print("✅ Rows:", len(merged))
