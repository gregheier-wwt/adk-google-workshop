"""Weather Agent using Google ADK and LiteLLM."""

import os
import sys
from typing import Any, Dict
import warnings
from dotenv import load_dotenv
import requests

# Suppress all internal library warnings cleanly across threads
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

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
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("VERTEXAI_PROJECT", "qwiklabs-gcp-01-763299e638c8")
os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("VERTEXAI_LOCATION", "global")


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
    """Convert an address or city name to geographic coordinates using the Google Maps Geocoding API.

    Args:
        address: Place name, city, or address to geocode.

    Returns:
        Dict containing latitude, longitude, and formatted address.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_MAPS_API_KEY is required for geocoding."}

    try:
        response = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address, "key": api_key},
            timeout=10,
        )
        if response.status_code != 200:
            return {"error": f"Google Maps Geocoding API failed with HTTP {response.status_code}"}

        data = response.json()
        if data.get("status") != "OK" or not data.get("results"):
            return {
                "error": f"Geocoding error: {data.get('status')} - {data.get('error_message', 'No results found')}"
            }

        location = data["results"][0]["geometry"]["location"]
        return {
            "latitude": float(location["lat"]),
            "longitude": float(location["lng"]),
            "formatted_address": data["results"][0]["formatted_address"],
        }
    except Exception as exc:
        return {"error": f"Geocoding request exception: {exc}"}


def create_weather_agent(model: str = "gemini-3.7-flash") -> LlmAgent:
    """Create an ADK LlmAgent supporting Gemini via Vertex AI and ADC.

    Args:
        model: Model identifier (e.g., 'gemini-3.7-flash').

    Returns:
        Configured LlmAgent instance.
    """
    return LlmAgent(
        name="weather_agent",
        model=model,
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
    print("=" * 65)
    print(" Google ADK Weather Agent - Challenge 1")
    print(f" Using Vertex AI Project: {os.getenv('GOOGLE_CLOUD_PROJECT')} (ADC Authenticated)")
    print(f" Region: {os.getenv('GOOGLE_CLOUD_LOCATION')}")
    print(" Sample inputs available in: sample_inputs.txt")
    print(" Type 'exit' or 'quit' to end session.")
    print("=" * 65)

    agent = create_weather_agent("gemini-3.7-flash")

    while True:
        try:
            user_input = input("\nEnter your question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting weather agent. Goodbye!")
                break

            response = run_agent(agent, user_input)
            print(f"\nAgent:\n{response}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting weather agent. Goodbye!")
            break
