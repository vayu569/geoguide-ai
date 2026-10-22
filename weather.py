mport requests

def get_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto"
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    return response.json()

if _name_ == "_main_":
    try:
        weather = get_weather(19.0760, 72.8777)
        print("GeoGuide AI - Current Weather")
        print(weather["current"])
    except requests.RequestException as error:
        print("Weather request failed:", error)
