# Gemini Instructions & Guidelines

Always adhere to the following project guidelines when generating, modifying, or executing code in this workspace:

## 1. Always Use the Google Agent Development Kit (Google ADK)
- Use `google-adk` (`from google.adk.agents.llm_agent import LlmAgent`, `from google.adk.runners import Runner`, `from google.adk.sessions import InMemorySessionService`, `from google.adk.agents.sequential_agent import SequentialAgent`, `vertexai.agent_engines`, etc.) for building all agents and multi-agent workflows.
- Maintain compatibility with ADK architecture, session services, runners, callbacks, tools, and deployment pipelines.

## 2. LLM Model Selection
- **Local & Standalone Agents (Challenges 1–4)**: Use **Gemini 3.7 Flash** (`gemini-3.7-flash`) with the `global` Vertex AI endpoint (`VERTEXAI_LOCATION=global`).
- **Google Agent Runtime Deployments (Challenges 5 & 6)**: Use **Gemini 2.5 Flash** (`gemini-2.5-flash`) for agents deployed remotely to Vertex AI Agent Runtime / Reasoning Engines (`us-central1`) to guarantee native regional compatibility.

## 3. Authentication & Environment Configuration
- Strictly rely on **Application Default Credentials (ADC)** and Vertex AI integration:
  ```python
  import os
  from dotenv import load_dotenv

  load_dotenv()
  os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
  os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("VERTEXAI_PROJECT", "")
  os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("VERTEXAI_LOCATION", "us-central1")
  ```
- Do not require or hardcode `GEMINI_API_KEY`.
- For Google Maps Geocoding, retrieve `GOOGLE_MAPS_API_KEY` from `.env`.
