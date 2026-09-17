"""
MCP server exposing a weather-forecast tool for Singapore, backed by the
free Open-Meteo API (no API key required).

Uses the v1 `FastMCP` API — paired with `pip install "mcp<2"` for
compatibility with langchain-mcp-adapters.

Run standalone for testing:
    mcp dev weather_server.py
"""

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("singapore-weather", dependencies=["requests"])

SINGAPORE_LAT = 1.3521
SINGAPORE_LON = 103.8198

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


@mcp.tool()
def get_singapore_forecast(days: int = 3) -> dict:
    """
    Get the weather forecast for Singapore for the next N days.

    Args:
        days: Number of days to forecast, between 1 and 7 (default 3).

    Returns:
        A dict with a per-day forecast: date, condition, max/min temp (C),
        and rain probability (%). Returns an "error" key if the request
        fails, instead of fabricated data.
    """
    days = max(1, min(days, 7))

    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": SINGAPORE_LAT,
                "longitude": SINGAPORE_LON,
                "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "Asia/Singapore",
                "forecast_days": days,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return {"error": f"Weather service unavailable: {e}"}

    try:
        daily = data["daily"]
        forecast = []
        for i, date in enumerate(daily["time"]):
            code = daily["weathercode"][i]
            forecast.append({
                "date": date,
                "condition": WEATHER_CODES.get(code, f"Unknown code {code}"),
                "max_temp_c": daily["temperature_2m_max"][i],
                "min_temp_c": daily["temperature_2m_min"][i],
                "rain_probability_percent": daily["precipitation_probability_max"][i],
            })
        return {"location": "Singapore", "forecast": forecast, "source": "Open-Meteo"}
    except (KeyError, IndexError) as e:
        return {"error": f"Unexpected response format from weather service: {e}"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
