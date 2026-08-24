"""Agent Workflow System using Google ADK.

Architecture:
- Greeter Agent: Welcomes the user and introduces the agent workflow.
- Weather Agent: Handles geocoding (Google Maps) and weather forecasts (NWS API).
- Answer Team (SequentialAgent Workflow):
    1. Search Agent: Finds data and facts to answer the question via Google Search.
    2. Critique Agent: Reviews initial response and makes constructive suggestions.
    3. Refine Agent: Rewrites the final response based on suggested improvements.
- Root Agent: Coordinating agent that routes user requests to greeter, weather, or the answer team.
- Callbacks: Logging to weather_agent.log & keyword validation blocking.
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
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

try:
    from google.adk.runners import _UNCACHED_TRANSFER_APPS
    _UNCACHED_TRANSFER_APPS.add("workflow_app")
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
# 5. Agent Workflow Setup (Greeter, Weather, Answer Team: Search -> Critique -> Refine)
# ============================================================================
def create_agent_workflow(
    model: str = "gemini-3.7-flash",
    blocked_keywords: Optional[List[str]] = None,
) -> LlmAgent:
    """Build and return an agent workflow system using Google ADK.

    Components:
    1. Greeter Agent: Welcomes users and introduces the multi-agent system.
    2. Weather Agent: Specialized sub-agent for US geocoding and NWS weather.
    3. Answer Team (SequentialAgent):
       a. Search Agent: Finds facts and data via Google Search tool.
       b. Critique Agent: Reviews the initial response and makes constructive suggestions.
       c. Refine Agent: Rewrites the response into a polished, complete final answer.
    4. Root Agent: Coordinating agent that routes user queries to the appropriate agent.
    """
    global BLOCKED_KEYWORDS
    if blocked_keywords is not None:
        BLOCKED_KEYWORDS = blocked_keywords

    server_side_config = types.GenerateContentConfig(
        tool_config=types.ToolConfig(include_server_side_tool_invocations=True)
    )

    # 1. Greeter Agent
    greeter_agent = LlmAgent(
        name="greeter_agent",
        model=model,
        description="Warmly welcomes the user and explains the capabilities of the agent workflow system.",
        instruction=(
            "You are a friendly greeter assistant. Welcome the user, introduce the available services "
            "(weather forecasts, web research with critique & refinement workflows), and invite questions."
        ),
        after_model_callback=log_model_response,
    )

    # 2. Weather Agent
    weather_agent = LlmAgent(
        name="weather_agent",
        model=model,
        description="Specialist sub-agent for retrieving US weather forecasts and geocoding locations.",
        instruction=(
            "You are a weather specialist assistant. When asked about weather in any location, "
            "first call geocode_address to get the coordinates, then call get_weather "
            "with those coordinates to retrieve the current forecast. Provide a clear summary."
        ),
        tools=[geocode_address, get_weather],
        after_model_callback=log_model_response,
    )

    # 3. Answer Team Components (Search -> Critique -> Refine)
    search_agent = LlmAgent(
        name="search_agent",
        model=model,
        description="Finds data and factual information to answer the question using Google Search.",
        instruction=(
            "You are a search researcher. Use the google_search tool to find accurate and up-to-date "
            "data and facts to answer the user's question clearly."
        ),
        tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
        generate_content_config=server_side_config,
        after_model_callback=log_model_response,
    )

    critique_agent = LlmAgent(
        name="critique_agent",
        model=model,
        description="Critiques the initial search findings and makes suggestions on how to improve the response.",
        instruction=(
            "You are a critical reviewer. Review the search agent's findings in the conversation. "
            "Evaluate accuracy, clarity, completeness, and structure. Make brief, constructive suggestions "
            "on how to improve the response."
        ),
        after_model_callback=log_model_response,
    )

    refine_agent = LlmAgent(
        name="refine_agent",
        model=model,
        description="Synthesizes and rewrites the final response based on the critique and suggestions.",
        instruction=(
            "You are a refinement specialist. Take the search findings and the critique suggestions, "
            "and write a polished, comprehensive, and well-structured final answer for the user."
        ),
        after_model_callback=log_model_response,
    )

    # 4. Sequential Answer Team
    answer_team = SequentialAgent(
        name="answer_team",
        description="A sequential workflow team that searches, critiques, and refines answers to questions.",
        sub_agents=[search_agent, critique_agent, refine_agent],
    )

    # 5. Root Coordinating Agent
    root_agent = LlmAgent(
        name="root_agent",
        model=model,
        description="Root coordinating agent that routes user queries to greeter, weather, or answer team.",
        instruction=(
            "You are the root coordinating agent. Direct incoming user requests:\n"
            "- If the user is greeting, saying hello, or asking what you can do, delegate to 'greeter_agent'.\n"
            "- If the query is about weather, temperatures, or forecasts, delegate to 'weather_agent'.\n"
            "- For all general questions, facts, news, research, or complex queries, delegate to 'answer_team'.\n"
            "Synthesize the response clearly and return the final answer to the user."
        ),
        sub_agents=[greeter_agent, weather_agent, answer_team],
        generate_content_config=server_side_config,
        before_model_callback=validate_and_log_user_prompt,
        after_model_callback=log_model_response,
    )

    return root_agent


# ============================================================================
# 6. Runner & Event Streaming Execution
# ============================================================================
def run_agent(agent: LlmAgent, prompt: str, show_events: bool = True) -> str:
    """Execute a prompt against the agent workflow, displaying delegation and stage events."""
    runner = Runner(
        app_name="workflow_app",
        agent=agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )
    message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    responses = []
    active_agents = set()
    final_output = []

    try:
        for event in runner.run(user_id="user", session_id="session", new_message=message):
            author = getattr(event, "author", None)
            if author and author not in ["user", "root_agent"]:
                active_agents.add(author)

            if event.message and event.message.parts:
                for part in event.message.parts:
                    if getattr(part, "text", None):
                        responses.append(part.text)
                        # Keep the latest response (e.g. from refine_agent or single sub-agent)
                        final_output = [part.text]
    except Exception as err:
        return f"[Runner Error: {err}]"

    if show_events and active_agents:
        delegated_to = " -> ".join(
            [a for a in ["greeter_agent", "weather_agent", "search_agent", "critique_agent", "refine_agent"] if a in active_agents]
        ) or ", ".join(sorted(active_agents))
        print(f"  [Workflow Event] Stages executed: {delegated_to}")

    # Return the refined final response or synthesized output
    return (final_output[0] if final_output else "\n".join(responses)).strip() or "No response from agent."


# ============================================================================
# 7. Interactive Entry Point
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Google ADK Agent Workflow - Challenge 4")
    print(" Workflow: Root -> [Greeter, Weather, Answer Team (Search -> Critique -> Refine)]")
    print(f" Blocked keywords: {', '.join(BLOCKED_KEYWORDS)}")
    print(f" Logs saved to: {LOG_FILE}")
    print(" Type 'exit' or 'quit' to end session.")
    print("=" * 65)

    if not os.getenv("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY is not set in .env")
        sys.exit(1)

    # Initialize the complete Agent Workflow System
    workflow_system = create_agent_workflow(model="gemini-3.7-flash")

    # Interactive REPL
    while True:
        try:
            user_input = input("\nEnter your question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting agent workflow. Goodbye!")
                break

            response = run_agent(workflow_system, user_input, show_events=True)
            print(f"\nAgent:\n{response}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting agent workflow. Goodbye!")
            break
