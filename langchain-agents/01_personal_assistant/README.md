# Demo 1: Personal Assistant with Memstate

This demo shows how to build a simple LangChain/LangGraph ReAct agent that remembers user preferences across multiple sessions using Memstate.

## What it demonstrates

1. **Persistent Memory**: The agent remembers facts from `session_1` and uses them in `session_2`.
2. **Auto-Extraction**: Using the `memstate_remember` tool, the agent can take natural language ("I prefer Python") and Memstate's custom LLMs will automatically extract it into structured keypaths (`users.alice.preferences.language = "Python"`).
3. **Updating Facts**: In `session_3`, the user changes their preference. Memstate updates the value but preserves the history.

## Running the demo

From the `langchain-agents` directory:

```bash
# Make sure your .env file is set up
python 01_personal_assistant/agent.py
```
