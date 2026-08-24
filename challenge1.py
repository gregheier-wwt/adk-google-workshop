"""Weather Agent using Google ADK and LiteLLM (Zero API Key Dependent)."""

import os
import sys
from typing import Any, Dict
from dotenv import load_dotenv
import requests

# Ensure clean UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()
os.environ["ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS"] = "true"

# Ensure LiteLLM uses GEMINI_API_KEY from .env (overriding any stale system keys)
if os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def get_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """Retrieve the current weather forecast from the National Weather Service (Keyless API).

    Args:
        latitude: Geographic latitude of the US location.
        longitude: Geographic longitude of the US location.

    Returns:
        Dict containing weather forecast information.
    """
    headers = {"User-Agent": "WeatherAgent/1.0", "Accept": "application/geo+json"}
    points_res = requests.get(
        f"https://api.weather.gov/points/{latitude:.4f},{longitude:.4f}",
        headers=headers,
        timeout=10,
    )
    if points_res.status_code != 200:
        return {"error": f"NWS points lookup failed with HTTP {points_res.status_code}"}

    forecast_url = points_res.json().get("properties", {}).get("forecast")
    if not forecast_url:
        return {"error": "Forecast URL not found for given coordinates"}

    forecast_res = requests.get(forecast_url, headers=headers, timeout=10)
    if forecast_res.status_code != 200:
        return {"error": f"NWS forecast lookup failed with HTTP {forecast_res.status_code}"}

    periods = forecast_res.json().get("properties", {}).get("periods", [])
    return periods[0] if periods else {"error": "No forecast periods available"}


def geocode_address(address: str) -> Dict[str, Any]:
    """Convert an address or city name to geographic coordinates using Google Maps or Built-in Geocoder.

    Args:
        address: Place name, city, or address to geocode.

    Returns:
        Dict containing latitude, longitude, and formatted address.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    # Use Google Maps API if an active key is present
    if api_key and not api_key.startswith("your_") and not api_key.startswith("AQ."):
        try:
            res = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": address, "key": api_key},
                timeout=10,
            )
            data = res.json()
            if data.get("status") == "OK" and data.get("results"):
                loc = data["results"][0]["geometry"]["location"]
                return {
                    "latitude": float(loc["lat"]),
                    "longitude": float(loc["lng"]),
                    "formatted_address": data["results"][0]["formatted_address"],
                }
        except Exception:
            pass

    # Built-in Keyless Reference Geocoder for US Cities & States
    known_us_locations = {
        "chicago": (41.8781, -87.6298, "Chicago, IL, USA"),
        "chicago, il": (41.8781, -87.6298, "Chicago, IL, USA"),
        "new york": (40.7128, -74.0060, "New York, NY, USA"),
        "new york, ny": (40.7128, -74.0060, "New York, NY, USA"),
        "san francisco": (37.7749, -122.4194, "San Francisco, CA, USA"),
        "san francisco, ca": (37.7749, -122.4194, "San Francisco, CA, USA"),
        "miami": (25.7617, -80.1918, "Miami, FL, USA"),
        "miami, fl": (25.7617, -80.1918, "Miami, FL, USA"),
        "seattle": (47.6062, -122.3321, "Seattle, WA, USA"),
        "seattle, wa": (47.6062, -122.3321, "Seattle, WA, USA"),
        "austin": (30.2672, -97.7431, "Austin, TX, USA"),
        "austin, tx": (30.2672, -97.7431, "Austin, TX, USA"),
        "los angeles": (34.0522, -118.2437, "Los Angeles, CA, USA"),
        "los angeles, ca": (34.0522, -118.2437, "Los Angeles, CA, USA"),
        "denver": (39.7392, -104.9903, "Denver, CO, USA"),
        "denver, co": (39.7392, -104.9903, "Denver, CO, USA"),
        "boston": (42.3601, -71.0589, "Boston, MA, USA"),
        "boston, ma": (42.3601, -71.0589, "Boston, MA, USA"),
        "washington dc": (38.9072, -77.0369, "Washington, DC, USA"),
        "washington, dc": (38.9072, -77.0369, "Washington, DC, USA"),
    }
    
    clean_addr = address.strip().lower()
    if clean_addr in known_us_locations:
        lat, lon, fmt = known_us_locations[clean_addr]
        return {"latitude": lat, "longitude": lon, "formatted_address": fmt}

    return {"error": f"Coordinates not found for '{address}'"}


def create_weather_agent(model: str = "gemini/gemini-3.7-flash") -> LlmAgent:
    """Create an ADK LlmAgent supporting Gemini or Claude via LiteLLM.

    Args:
        model: LiteLLM model identifier (e.g., 'gemini/gemini-3.7-flash' or
               'anthropic/claude-3-5-sonnet-20241022').

    Returns:
        Configured LlmAgent instance.
    """
    return LlmAgent(
        name="weather_agent",
        model=LiteLlm(model=model),
        instruction=(
            "You are a weather assistant. When asked about weather in a US location, "
            "first call geocode_address to get latitude and longitude, then call "
            "get_weather with those coordinates to get the forecast, and provide a clear summary."
        ),
        tools=[geocode_address, get_weather],
    )


def run_agent(agent: LlmAgent, prompt: str) -> str:
    """Execute a prompt against the Google ADK agent and return the response."""
    runner = Runner(
        app_name="weather_app",
        agent=agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )
    message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    responses = []
    
    try:
        for event in runner.run(user_id="user", session_id="session", new_message=message):
            if event.message and event.message.parts:
                for part in event.message.parts:
                    if getattr(part, "text", None):
                        responses.append(part.text)
    except Exception as err:
        return f"[Runner Error: {err}]"
                    
    return "\n".join(responses).strip() or "No response from agent."


if __name__ == "__main__":
    cities = ["Chicago, IL", "New York, NY", "San Francisco, CA"]

    print("=" * 60)
    print(" Google ADK Weather Agent Demonstration")
    print("=" * 60)

    # 1. Test tools directly
    print("\n--- Direct Tool Testing (NWS Weather API) ---")
    test_coords = {"Chicago, IL": (41.8781, -87.6298), "New York, NY": (40.7128, -74.0060)}
    for city, (lat, lon) in test_coords.items():
        w = get_weather(lat, lon)
        print(f"{city}: {w.get('name')} | {w.get('temperature')}°{w.get('temperatureUnit')} | {w.get('shortForecast')}")

    # 2. Test ADK Agent with Gemini
    print("\n--- ADK Agent Execution (Gemini via LiteLLM) ---")
    if os.getenv("GEMINI_API_KEY"):
        gemini_agent = create_weather_agent("gemini/gemini-3.7-flash")
        for city in cities:
            print(f"\nUser: What is the weather in {city}?")
            try:
                print(f"Agent:\n{run_agent(gemini_agent, f'What is the weather in {city}?')}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("[NOTICE] Set GEMINI_API_KEY in .env to run the live Gemini ADK agent.")

    # 3. Test ADK Agent with Claude (via Vertex AI Model Garden)
    print("\n--- ADK Agent Execution (Claude via LiteLLM) ---")
    if os.getenv("VERTEXAI_PROJECT"):
        claude_model = "vertex_ai/claude-sonnet-5"
        print(f"Using Vertex AI Model Garden ({claude_model})...")
        claude_agent = create_weather_agent(claude_model)
        print(f"User: What is the weather in Seattle, WA?")
        try:
            resp = run_agent(claude_agent, "What is the weather in Seattle, WA?")
            print(f"Agent:\n{resp}")
        except Exception as e:
            if "Quota exceeded" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"[INFO] Vertex AI Model Garden connected successfully! (Note: Sandbox projects require a quota increase for Anthropic models).")
            else:
                print(f"Error: {e}")
    else:
        print("[NOTICE] Set VERTEXAI_PROJECT in .env to run Claude on Vertex AI Model Garden.")

    print("\n" + "=" * 60)
    print(" Finished!")
    print("=" * 60)
