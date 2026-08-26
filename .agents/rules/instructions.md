# Workspace Rules: Google ADK & Gemini 3.7 Flash

- **Always use the Google ADK (`google-adk`)**: All agents, runners, multi-agent orchestrations, sequential agents, session management, and deployment code must use Google ADK.
- **LLM Model**:
  - Challenges 1–4 (Local & Standalone): Use `gemini-3.7-flash` on the `global` Vertex AI endpoint.
  - Challenges 5 & 6 (Agent Runtime Deployments): Use `gemini-2.5-flash` for native regional compatibility in `us-central1`.
- **Authentication**: Rely strictly on Application Default Credentials (ADC) with Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI="1"`). Do not use or require a Gemini API key.
