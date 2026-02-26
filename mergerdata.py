import requests
import pandas as pd

lat = 13.0827
lon = 80.2707  # Chennai

start_date = "2022-01-01"
end_date   = "2024-10-01"

url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={lat}&longitude={lon}"
    f"&start_date={start_date}&end_date={end_date}"
    f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
    f"&timezone=auto"
)

print("Fetching:", url)

resp = requests.get(url).json()

# ✅ Check validity
if "daily" not in resp:
    print("❌ ERROR: API did not return daily data")
    print(resp)
    raise SystemExit

df_weather = pd.DataFrame({
    "date": resp["daily"]["time"],
    "temp_max": resp["daily"]["temperature_2m_max"],
    "temp_min": resp["daily"]["temperature_2m_min"],
    "rain_mm": resp["daily"]["precipitation_sum"],
    "wind_kmh": resp["daily"]["windspeed_10m_max"]
})

df_weather.to_csv("weather.csv", index=False)
print("✅ Weather data saved as weather.csv")
