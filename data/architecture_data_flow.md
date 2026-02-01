# Architecture - Data Flow
- User submits request to /agent.
- Router selects a tool and retrieves context from Qdrant.
- LLM generates response and returns structured output.
