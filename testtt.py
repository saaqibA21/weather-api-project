import requests

API_KEY = "60dc63cda3a14addb0f7cd93f8f19641"
city = "London"

url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

resp = requests.get(url)
print("Status:", resp.status_code)
print(resp.json())
