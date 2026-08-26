"""Agent Workflow Deployment to Google Agent Runtime (Vertex AI Agent Engine) using Google ADK - Challenge 5

Challenge 5:
- Uses vertexai.agent_engines.create() to deploy the multi-agent workflow to Google's Agent Runtime.
- Implements comprehensive session management: create_session, list_sessions, get_session, delete_session.
- Implements request streaming and event handling via stream_query().

Architecture:
- Greeter Agent: Welcomes user and introduces workflow services.
- Weather Agent: Specialized for US geocoding (Google Maps) and weather forecasts (NWS API).
- Answer Team (SequentialAgent):
    1. Search Agent: Finds facts & data via Google Search.
    2. Critique Agent: Reviews search findings and makes constructive suggestions.
    3. Refine Agent: Rewrites the response into a polished, complete final answer.
- Root Agent: Coordinating supervisor agent that delegates requests to appropriate agents.
- Callbacks: Logging to weather_agent.log & keyword validation blocking.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union
import warnings
from dotenv import load_dotenv
import requests

# Suppress all internal library warnings and background notices cleanly
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

for log_name in ["google", "google.adk", "google.genai", "google_genai", "google_genai.models", "urllib3", "google.cloud"]:
    _log = logging.getLogger(log_name)
    _log.setLevel(logging.CRITICAL)
    _log.propagate = False

# Ensure clean UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import vertexai
from vertexai import agent_engines
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.genai import types

try:
    from google.adk.runners import _UNCACHED_TRANSFER_APPS
    _UNCACHED_TRANSFER_APPS.add("workflow_app")
    _UNCACHED_TRANSFER_APPS.add("default_app_name")
except Exception:
    pass

load_dotenv()
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("VERTEXAI_PROJECT")
os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("VERTEXAI_LOCATION", "global")


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
    model: str = "gemini-2.5-flash",
    blocked_keywords: Optional[List[str]] = None,
) -> LlmAgent:
    """Build and return an agent workflow system using Google ADK.

    Components:
    1. Greeter Agent: Welcomes users and introduces the multi-agent system.
    2. Weather Agent: Specialist sub-agent for US geocoding and NWS weather.
    3. Answer Team (SequentialAgent):
       a. Search Agent: Finds facts and data via Google Search tool.
       b. Critique Agent: Reviews the initial response and makes constructive suggestions.
       c. Refine Agent: Rewrites the response into a polished, complete final answer.
    4. Root Agent: Coordinating supervisor that routes queries to greeter, weather, or answer team.
    """
    global BLOCKED_KEYWORDS
    if blocked_keywords is not None:
        BLOCKED_KEYWORDS = blocked_keywords

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
        tools=[GoogleSearchTool()],
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
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
        before_model_callback=validate_and_log_user_prompt,
        after_model_callback=log_model_response,
    )

    return root_agent


# ============================================================================
# 6. Google Agent Runtime Deployment (agent_engines.create)
# ============================================================================
def init_vertexai(
    project: Optional[str] = None,
    location: Optional[str] = None,
    staging_bucket: Optional[str] = None,
) -> None:
    """Initialize Vertex AI configuration for Agent Engine deployment."""
    project = project or os.getenv("VERTEXAI_PROJECT")
    # Vertex AI Reasoning Engines require a regional endpoint (e.g. us-central1)
    location = location or os.getenv("AGENT_ENGINE_LOCATION", "us-central1")
    staging_bucket = staging_bucket or os.getenv("VERTEXAI_STAGING_BUCKET")
    if not staging_bucket and project:
        staging_bucket = f"gs://{project}-staging"

    kwargs: Dict[str, Any] = {"location": location}
    if project:
        kwargs["project"] = project
    if staging_bucket:
        if not staging_bucket.startswith("gs://"):
            staging_bucket = f"gs://{staging_bucket}"
        kwargs["staging_bucket"] = staging_bucket

    vertexai.init(**kwargs)
    print(f"  [Vertex AI] Initialized (Project: {project or 'default'}, Location: {location}, Staging Bucket: {staging_bucket or 'None'})")


def deploy_to_agent_runtime(
    agent: LlmAgent,
    display_name: str = "weather-agent-workflow",
    description: str = "ADK Multi-Agent Workflow on Google Agent Runtime",
    requirements: Optional[List[str]] = None,
    env_vars: Optional[Dict[str, str]] = None,
    staging_bucket: Optional[str] = None,
) -> agent_engines.AgentEngine:
    """Deploy the ADK agent workflow to Google's Agent Runtime using agent_engines.create().

    Args:
        agent: The root ADK LlmAgent workflow to deploy.
        display_name: Human-readable display name for the deployed agent engine.
        description: Description of the deployed agent engine.
        requirements: List of PyPI dependencies required by the agent.
        env_vars: Environment variables to make available to the deployed agent.
        staging_bucket: Cloud Storage staging bucket (gs://...).

    Returns:
        The deployed remote AgentEngine instance.
    """
    if requirements is None:
        requirements = [
            "google-cloud-aiplatform[agent_engines,adk]",
            "requests",
            "python-dotenv",
        ]

    if env_vars is None:
        env_vars = {}
        for key in ["GOOGLE_MAPS_API_KEY", "VERTEXAI_PROJECT", "GOOGLE_GENAI_USE_VERTEXAI"]:
            val = os.getenv(key)
            if val:
                env_vars[key] = val
        # Ensure Gemini model calls use us-central1 location inside container
        env_vars["VERTEXAI_LOCATION"] = "us-central1"

    print("\n" + "=" * 65)
    print(f" Deploying agent workflow to Google Agent Runtime...")
    print(f" Display Name: {display_name}")
    print(f" Requirements: {', '.join(requirements)}")
    print(f" Environment Variables: {list(env_vars.keys())}")
    print("=" * 65)

    # Use agent_engines.create() to package and deploy the agent to Google Agent Runtime
    remote_agent = agent_engines.create(
        agent_engine=agent,
        requirements=requirements,
        display_name=display_name,
        description=description,
        env_vars=env_vars,
    )

    print(f"\n[SUCCESS] Agent deployed to Google Agent Runtime!")
    print(f"Resource Name: {remote_agent.resource_name}")
    return remote_agent


def get_deployed_agent(resource_name: str) -> agent_engines.AgentEngine:
    """Retrieve an existing deployed Agent Engine instance from Google Agent Runtime.

    Args:
        resource_name: Full resource name or ID of the deployed Agent Engine.

    Returns:
        AgentEngine instance connected to the deployed remote agent.
    """
    print(f" Connecting to deployed Agent Engine: {resource_name}...")
    return agent_engines.get(resource_name)


# ============================================================================
# 7. Session Management Lifecycle on Agent Runtime
# ============================================================================
def create_session(engine: Any, user_id: str = "user_workshop_1") -> Dict[str, Any]:
    """Create a new stateful session for a user on the Agent Engine.

    Args:
        engine: The deployed remote AgentEngine or local AdkApp runtime.
        user_id: Unique user identifier for session tracking.

    Returns:
        Dict representing session with 'id'.
    """
    session = engine.create_session(user_id=user_id)
    session_id = session.get("id") if isinstance(session, dict) else getattr(session, "id", str(session))
    print(f"  [Session Management] Created session '{session_id}' for user '{user_id}'")
    return {"id": session_id, "user_id": user_id}


def list_sessions(engine: Any, user_id: str = "user_workshop_1") -> List[Any]:
    """List all active sessions for a user on the Agent Engine.

    Args:
        engine: The deployed remote AgentEngine or local AdkApp runtime.
        user_id: Unique user identifier for session tracking.

    Returns:
        List of session objects/dictionaries.
    """
    try:
        sessions = list(engine.list_sessions(user_id=user_id))
        return sessions
    except Exception as exc:
        print(f"  [Session Management] Error listing sessions: {exc}")
        return []


def delete_session(engine: Any, session_id: str, user_id: str = "user_workshop_1") -> bool:
    """Delete an existing session on the Agent Engine.

    Args:
        engine: The deployed remote AgentEngine or local AdkApp runtime.
        session_id: Session identifier to delete.
        user_id: Unique user identifier.

    Returns:
        True if deleted successfully, False otherwise.
    """
    try:
        engine.delete_session(session_id=session_id, user_id=user_id)
        print(f"  [Session Management] Deleted session '{session_id}'")
        return True
    except Exception as exc:
        print(f"  [Session Management] Error deleting session '{session_id}': {exc}")
        return False


# ============================================================================
# 8. Querying Deployed Agent with Streaming & Event Processing
# ============================================================================
def query_agent_runtime(
    agent_engine: Any,
    prompt: str,
    user_id: str = "user_workshop_1",
    session_id: Optional[str] = None,
    show_events: bool = True,
) -> str:
    """Send a message to the Agent Engine and stream the response events.

    Args:
        agent_engine: Remote AgentEngine or local AdkApp instance.
        prompt: The user's query text.
        user_id: User identifier.
        session_id: Optional session identifier for maintaining conversation history.
        show_events: If True, displays workflow execution stages.

    Returns:
        The aggregated final response string.
    """
    kwargs: Dict[str, Any] = {"user_id": user_id, "message": prompt}
    if session_id:
        kwargs["session_id"] = session_id

    responses = []
    executed_stages = []
    final_output = []

    try:
        stream = agent_engine.stream_query(**kwargs)
        for event in stream:
            # Handle dictionary events (standard Agent Engine JSON event schema)
            if isinstance(event, dict):
                # Track agent stage execution events
                author = event.get("author") or event.get("agent_name")
                if author and author not in ["user", "root_agent"]:
                    executed_stages.append(author)

                # Extract text content from parts
                content = event.get("content", {})
                parts = content.get("parts", []) if isinstance(content, dict) else []
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        responses.append(part["text"])
                    elif hasattr(part, "text") and part.text:
                        responses.append(part.text)

                # Fallback for top-level text/message
                if "text" in event and event["text"]:
                    responses.append(str(event["text"]))
                if "output" in event and event["output"]:
                    final_output.append(str(event["output"]))

            # Handle object events
            elif hasattr(event, "content"):
                content = getattr(event, "content", None)
                if content and hasattr(content, "parts"):
                    for part in content.parts:
                        if getattr(part, "text", None):
                            responses.append(part.text)
            elif isinstance(event, str):
                responses.append(event)
            else:
                responses.append(str(event))

    except Exception as exc:
        return f"[Agent Runtime Error: {exc}]"

    if show_events and executed_stages:
        unique_stages = list(dict.fromkeys(executed_stages))
        print(f"  [Runtime Event] Stages executed: {' -> '.join(unique_stages)}")

    return (final_output[0] if final_output else "\n".join(responses)).strip() or "No response received from Agent Runtime."


# ============================================================================
# 9. Interactive CLI Entry Point
# ============================================================================
def main():
    print("=" * 70)
    print(" Google ADK Agent Workflow - Challenge 5")
    print(" Deploy to Google's Agent Runtime using agent_engines.create()")
    print(" Multi-Agent Workflow: Root -> [Greeter, Weather, Answer Team (Search->Critique->Refine)]")
    print(f" Blocked keywords: {', '.join(BLOCKED_KEYWORDS)}")
    print(f" Logs saved to: {LOG_FILE}")
    print("=" * 70)

    # Initialize Vertex AI for Agent Engine Deployment
    init_vertexai(location="us-central1")

    # Build the complete Agent Workflow System
    workflow_agent = create_agent_workflow(model="gemini-2.5-flash")

    # Mode Selection: Deploy remote agent, connect to existing, or run local AdkApp
    print("\nSelect Deployment / Execution Mode:")
    print("  [1] Deploy new Agent Engine with agent_engines.create() (Remote)")
    print("  [2] Connect to existing deployed Agent Engine (Remote)")
    print("  [3] Run with local AdkApp runtime (Local Testing)")

    mode_choice = input("\nEnter choice [1/2/3] (default 3): ").strip() or "3"

    active_engine = None
    user_id = "user_workshop_1"

    if mode_choice == "1":
        staging_bucket = os.getenv("VERTEXAI_STAGING_BUCKET") or f"gs://{os.getenv('VERTEXAI_PROJECT', 'default')}-staging"
        init_vertexai(location="us-central1", staging_bucket=staging_bucket)

        try:
            active_engine = deploy_to_agent_runtime(
                agent=workflow_agent,
                display_name="weather-agent-workflow",
                description="ADK Multi-Agent Workflow on Google Agent Runtime",
                requirements=["google-cloud-aiplatform[agent_engines,adk]", "requests", "python-dotenv"],
            )
        except Exception as exc:
            print(f"\n[Deployment Failed] {exc}")
            print("Falling back to local AdkApp runtime...")
            active_engine = agent_engines.AdkApp(agent=workflow_agent)
    elif mode_choice == "2":
        resource_name = input("Enter deployed Agent Engine resource name: ").strip()
        if resource_name:
            try:
                active_engine = get_deployed_agent(resource_name)
            except Exception as exc:
                print(f"[Error] Failed to fetch agent: {exc}")
                print("Falling back to local AdkApp runtime...")
                active_engine = agent_engines.AdkApp(agent=workflow_agent)
        else:
            print("No resource name provided. Using local AdkApp runtime.")
            active_engine = agent_engines.AdkApp(agent=workflow_agent)
    else:
        print("\nInitializing local AdkApp runtime for testing...")
        active_engine = agent_engines.AdkApp(agent=workflow_agent)

    # Manage Initial Session
    print("\n--- Session Management ---")
    current_session = create_session(active_engine, user_id=user_id)
    session_id = current_session.get("id")

    print("\nCommands:")
    print("  - Type any question or message to query the agent.")
    print("  - '/new'      : Create a new session")
    print("  - '/sessions' : List all active sessions")
    print("  - '/delete'   : Delete current session and start a new one")
    print("  - 'exit'/'quit': Clean up and exit")
    print("-" * 70)

    # Interactive REPL
    while True:
        try:
            user_input = input(f"\n[Session: {session_id[:8]}...] Enter query: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print(f"\nCleaning up session '{session_id}'...")
                delete_session(active_engine, session_id=session_id, user_id=user_id)
                print("Exiting Agent Runtime. Goodbye!")
                break

            if user_input.lower() == "/new":
                current_session = create_session(active_engine, user_id=user_id)
                session_id = current_session.get("id")
                continue

            if user_input.lower() == "/sessions":
                sessions = list_sessions(active_engine, user_id=user_id)
                print(f"Active sessions for '{user_id}':")
                for s in sessions:
                    s_id = s.get("id") if isinstance(s, dict) else getattr(s, "id", str(s))
                    prefix = "-> " if s_id == session_id else "   "
                    print(f"  {prefix}{s_id}")
                continue

            if user_input.lower() == "/delete":
                delete_session(active_engine, session_id=session_id, user_id=user_id)
                current_session = create_session(active_engine, user_id=user_id)
                session_id = current_session.get("id")
                continue

            response = query_agent_runtime(
                agent_engine=active_engine,
                prompt=user_input,
                user_id=user_id,
                session_id=session_id,
                show_events=True,
            )
            print(f"\nAgent Runtime Response:\n{response}")

        except (KeyboardInterrupt, EOFError):
            print(f"\nCleaning up session '{session_id}'...")
            delete_session(active_engine, session_id=session_id, user_id=user_id)
            print("\nExiting Agent Runtime. Goodbye!")
            break


if __name__ == "__main__":
    main()
