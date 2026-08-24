"""Weather Agent with ADK Callbacks for Logging and Validation"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional
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
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()
os.environ.pop("GOOGLE_API_KEY", None)

# ============================================================================
# 1. Logging Setup (writes everything to weather_agent.log)
# ============================================================================
LOG_FILE = "weather_agent.log"

logger = logging.getLogger("weather_agent")
logger.setLevel(logging.INFO)

# Create file handler for weather_agent.log
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)

# Avoid duplicate handlers on re-runs
if not logger.handlers:
    logger.addHandler(file_handler)

# ============================================================================
# 2. Configurable List of Blocked Keywords for Input Validation
# ============================================================================
BLOCKED_KEYWORDS: List[str] = ["hack", "secret", "password", "confidential", "forbidden", "override"]


# ============================================================================
# 3. Callback Functions (Validation & Logging)
# ============================================================================
def validate_and_log_user_prompt(callback_context: Any, llm_request: Any) -> Optional[LlmResponse]:
    """Callback function to validate user input and log user prompts before sending to the model.

    If any forbidden keyword is found, intercepts the request and blocks the call to the model.
    """
    user_prompt = ""
    if hasattr(llm_request, "contents") and llm_request.contents:
        for content in llm_request.contents:
            if hasattr(content, "parts") and content.parts:
                for part in content.parts:
                    if getattr(part, "text", None):
                        user_prompt += part.text + " "

    user_prompt = user_prompt.strip()

    # Requirement 1: Log user prompt
    logger.info(f"USER PROMPT: {user_prompt}")

    # Requirement 3: Check and validate user input against blocked keywords
    for keyword in BLOCKED_KEYWORDS:
        if keyword.lower() in user_prompt.lower():
            warning_msg = f"SECURITY ALERT: Blocked keyword '{keyword}' detected in input: '{user_prompt}'"
            logger.warning(warning_msg)

            # Block request by returning early LlmResponse (prevents sending to model)
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=f"[BLOCKED BY VALIDATION CALLBACK] Your request was blocked before reaching the model because it contains the forbidden keyword: '{keyword}'."
                        )
                    ],
                )
            )

    return None


def log_model_response(callback_context: Any, llm_response: Any) -> Optional[LlmResponse]:
    """Callback function to log model responses after generation."""
    resp_text = ""
    if hasattr(llm_response, "content") and llm_response.content:
        if hasattr(llm_response.content, "parts") and llm_response.content.parts:
            for part in llm_response.content.parts:
                if getattr(part, "text", None):
                    resp_text += part.text

    # Requirement 2: Log model response
    logger.info(f"MODEL RESPONSE: {resp_text.strip()}")
    return None


# ============================================================================
# 4. Tools (PEP 8 Compliant NWS & Google Maps Geocoding)
# ============================================================================
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


# ============================================================================
# 5. Agent Factory & Execution
# ============================================================================
def create_weather_agent(
    model: str = "gemini-3.7-flash",
    blocked_keywords: Optional[List[str]] = None,
) -> LlmAgent:
    """Create a native Google ADK LlmAgent configured with logging and validation callbacks.

    Args:
        model: Native Gemini model identifier (e.g., 'gemini-3.7-flash').
        blocked_keywords: Optional list of keywords to block prior to sending to model.

    Returns:
        Configured LlmAgent instance.
    """
    global BLOCKED_KEYWORDS
    if blocked_keywords is not None:
        BLOCKED_KEYWORDS = blocked_keywords

    return LlmAgent(
        name="weather_agent",
        model=model,
        instruction=(
            "You are a weather assistant. When asked about weather in a US location, "
            "first call geocode_address to get latitude and longitude, then call "
            "get_weather with those coordinates to get the forecast, and provide a clear summary."
        ),
        tools=[geocode_address, get_weather],
        before_model_callback=validate_and_log_user_prompt,
        after_model_callback=log_model_response,
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


# ============================================================================
# 6. Interactive User Prompt Mode
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Google ADK Weather Agent")
    print(f" Blocked keywords: {', '.join(BLOCKED_KEYWORDS)}")
    print(f" Logs saved to: {LOG_FILE}")
    print(" Type 'exit' or 'quit' to end session.")
    print("=" * 65)

    if not os.getenv("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY is not set in .env")
        sys.exit(1)

    # Initialize agent with validation & logging callbacks
    agent = create_weather_agent(
        model="gemini-3.7-flash",
        blocked_keywords=["hack", "secret", "password", "confidential", "forbidden"],
    )

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
