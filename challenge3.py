"""Multi-Agent System using Google ADK

Architecture:
- Root Agent (Coordinating Agent): Receives user requests and delegates to sub-agents.
- Weather Agent (Sub-Agent): Queries Google Maps Geocoding API and National Weather Service API.
- Search Agent (Sub-Agent): Uses ADK built-in Google Search tool for web queries and general knowledge.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional
import warnings
from dotenv import load_dotenv
import requests

# Suppress all internal library warnings and background notices cleanly
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

for log_name in ["google", "google.adk", "google.genai", "google_genai", "google_genai.models", "urllib3"]:
    _log = logging.getLogger(log_name)
    _log.setLevel(logging.CRITICAL)
    _log.propagate = False

# Ensure clean UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

try:
    from google.adk.runners import _UNCACHED_TRANSFER_APPS
    _UNCACHED_TRANSFER_APPS.add("multi_agent_system")
except Exception:
    pass

load_dotenv()
os.environ.pop("GOOGLE_API_KEY", None)

# ============================================================================
# 1. Logging Setup (writes everything to weather_agent.log)
# ============================================================================
LOG_FILE = "weather_agent.log"

logger = logging.getLogger("weather_agent")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)

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
    """Callback function to validate user input and log user prompts before sending to the model."""
    user_prompt = ""
    if hasattr(llm_request, "contents") and llm_request.contents:
        for content in llm_request.contents:
            if hasattr(content, "parts") and content.parts:
                for part in content.parts:
                    if getattr(part, "text", None):
                        user_prompt += part.text + " "

    user_prompt = user_prompt.strip()

    # Log user prompt
    if user_prompt:
        logger.info(f"USER PROMPT: {user_prompt}")

    # Check and validate user input against blocked keywords
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

    if resp_text.strip():
        logger.info(f"MODEL RESPONSE: {resp_text.strip()}")
    return None


# ============================================================================
# 4. Custom Tools for Weather Agent
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
# 5. Multi-Agent System Setup (Weather Agent, Search Agent, Root Agent)
# ============================================================================
def create_multi_agent_system(
    model: str = "gemini-3.7-flash",
    blocked_keywords: Optional[List[str]] = None,
) -> LlmAgent:
    """Build and return a multi-agent system with a root agent and two specialized sub-agents.

    - Sub-Agent 1: Weather Agent (handles geocoding and NWS weather forecasts)
    - Sub-Agent 2: Search Agent (handles web search and general knowledge via Google Search tool)
    - Root Agent: Coordinates and delegates tasks to the appropriate sub-agent.
    """
    global BLOCKED_KEYWORDS
    if blocked_keywords is not None:
        BLOCKED_KEYWORDS = blocked_keywords

    # Shared server-side tool config for Google Search compatibility
    server_side_config = types.GenerateContentConfig(
        tool_config=types.ToolConfig(include_server_side_tool_invocations=True)
    )

    # 1. Weather Agent Sub-Agent
    weather_agent = LlmAgent(
        name="weather_agent",
        model=model,
        description="Specialist sub-agent for retrieving US weather forecasts, conditions, and geocoding locations.",
        instruction=(
            "You are a weather specialist assistant. When asked about weather in any location, "
            "first call geocode_address to get the geographic coordinates, then call get_weather "
            "with those coordinates to retrieve the current forecast. Provide a clear, detailed summary."
        ),
        tools=[geocode_address, get_weather],
        after_model_callback=log_model_response,
    )

    # 2. Search Agent Sub-Agent (uses ADK built-in Google Search tool)
    search_agent = LlmAgent(
        name="search_agent",
        model=model,
        description="Specialist sub-agent for web searches, general knowledge, current events, facts, and news using Google Search.",
        instruction=(
            "You are a web search specialist assistant. Use the google_search tool to find accurate, "
            "up-to-date information, facts, news, and answers to general questions."
        ),
        tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
        generate_content_config=server_side_config,
        after_model_callback=log_model_response,
    )

    # 3. Root Agent (Coordinating Agent) with Input Validation & Logging Callbacks
    root_agent = LlmAgent(
        name="root_agent",
        model=model,
        description="Root coordinating agent that analyzes user requests and delegates them to specialized sub-agents.",
        instruction=(
            "You are the root coordinating assistant. Analyze the incoming user request and delegate it to the best sub-agent:\n"
            "- If the query is about weather, temperatures, forecasts, or meteorological conditions, delegate to 'weather_agent'.\n"
            "- If the query is about current events, news, general facts, web knowledge, or topics other than weather, delegate to 'search_agent'.\n"
            "Synthesize the response clearly and return the final answer to the user."
        ),
        sub_agents=[weather_agent, search_agent],
        generate_content_config=server_side_config,
        before_model_callback=validate_and_log_user_prompt,
        after_model_callback=log_model_response,
    )

    return root_agent


# ============================================================================
# 3. Runner & Event Streaming Execution
# ============================================================================
def run_agent(agent: LlmAgent, prompt: str, show_events: bool = True) -> str:
    """Execute a prompt against the multi-agent system, displaying delegation events."""
    runner = Runner(
        app_name="multi_agent_system",
        agent=agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )
    message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    responses = []
    active_agents = set()

    try:
        for event in runner.run(user_id="user", session_id="session", new_message=message):
            author = getattr(event, "author", None)
            if author and author not in ["user", "root_agent"]:
                active_agents.add(author)

            if event.message and event.message.parts:
                for part in event.message.parts:
                    if getattr(part, "text", None):
                        responses.append(part.text)
    except Exception as err:
        return f"[Runner Error: {err}]"

    if show_events and active_agents:
        delegated_to = ", ".join(sorted(active_agents))
        print(f"  [Delegation Event] Delegated task to sub-agent: {delegated_to}")

    return "\n".join(responses).strip() or "No response from agent."


# ============================================================================
# 4. Interactive & Demonstration Entry Point
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Google ADK Multi-Agent System - Challenge 3")
    print(" Root Agent coordinating: [weather_agent, search_agent]")
    print(f" Blocked keywords: {', '.join(BLOCKED_KEYWORDS)}")
    print(f" Logs saved to: {LOG_FILE}")
    print(" Type 'exit' or 'quit' to end session.")
    print("=" * 65)

    if not os.getenv("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY is not set in .env")
        sys.exit(1)

    # Initialize the complete Multi-Agent System
    root_agent = create_multi_agent_system(model="gemini-3.7-flash")

    # Interactive REPL
    while True:
        try:
            user_input = input("\nEnter your question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting multi-agent system. Goodbye!")
                break

            response = run_agent(root_agent, user_input, show_events=True)
            print(f"\nAgent:\n{response}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting multi-agent system. Goodbye!")
            break
