import requests

API_KEY = "60dc63cda3a14addb0f7cd93f8f19641"

def get_weather(city):
    """
    Fetches live weather using OpenWeather API and converts it
    into your ML model’s expected inputs.
    """

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()

    if "main" not in data:
        raise ValueError("Invalid city name or API failure")

    temp_max = data["main"]["temp_max"]
    temp_min = data["main"]["temp_min"]
    wind_kmh = data["wind"]["speed"] * 3.6   # convert m/s → km/h

    # Rain last 3 hours
    rain_mm = data.get("rain", {}).get("3h", 0.0)

    return {
        "temp_max": temp_max,
        "temp_min": temp_min,
        "rain_mm": rain_mm,
        "wind_kmh": wind_kmh
    }
