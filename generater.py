import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

START_DATE = datetime(2021, 1, 1)
DAYS = 1000

records = []
rain_history = []

for i in range(DAYS):
    date = START_DATE + timedelta(days=i)

    temp_max = np.random.uniform(25, 38)
    temp_min = temp_max - np.random.uniform(4, 10)

    rain_mm = max(0, np.random.gamma(2, 3))
    rain_history.append(rain_mm)

    wind_kmh = np.random.uniform(2, 20)
    humidity = np.random.uniform(55, 95)
    pressure = np.random.uniform(990, 1025)
    cloud_cover = np.random.uniform(20, 100)

    dew_point = temp_min + np.random.uniform(1, 4)
    temp_range = temp_max - temp_min

    heat_index = temp_max + 0.33 * humidity - 0.7 * wind_kmh - 4

    rain_3day_avg = np.mean(rain_history[-3:])
    rain_7day_sum = np.sum(rain_history[-7:])

    month = date.month
    day_of_year = date.timetuple().tm_yday

    # ---- Disease Logic ----
    if rain_mm > 15 and humidity > 75:
        disease = np.random.choice(["dengue", "chikungunya"])
    elif temp_max > 30 and humidity > 65:
        disease = "malaria"
    elif rain_7day_sum > 50:
        disease = "cholera"
    else:
        disease = np.random.choice(["dengue", "malaria", "chikungunya"])

    records.append([
        date.strftime("%Y-%m-%d"),
        round(temp_max, 2),
        round(temp_min, 2),
        round(rain_mm, 2),
        round(wind_kmh, 2),
        round(humidity, 2),
        round(pressure, 2),
        round(cloud_cover, 2),
        round(dew_point, 2),
        round(temp_range, 2),
        round(heat_index, 2),
        round(rain_3day_avg, 2),
        round(rain_7day_sum, 2),
        month,
        day_of_year,
        disease
    ])

columns = [
    "date", "temp_max", "temp_min", "rain_mm", "wind_kmh",
    "humidity", "pressure", "cloud_cover", "dew_point",
    "temp_range", "heat_index", "rain_3day_avg", "rain_7day_sum",
    "month", "day_of_year", "disease"
]

df = pd.DataFrame(records, columns=columns)
df.to_csv("weather_disease_dataset_1000_days.csv", index=False)

print("✅ Dataset generated: weather_disease_dataset_1000_days.csv")
