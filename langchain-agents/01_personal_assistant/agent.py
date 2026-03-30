"""
Demo 1: Personal Assistant with Persistent Memory
==================================================
Shows how to build a LangGraph ReAct agent that remembers user preferences
across multiple sessions using Memstate.

Key concepts:
- MemstateStore: drop-in LangGraph BaseStore with versioned, semantic memory
- get_memstate_tools: gives the agent tools to read/write memory explicitly
- memstate_remember: auto-extracts structured facts from natural language
  (async, uses LLM extraction — best for unstructured text)
- memstate_store: direct synchronous keypath write — best for precise facts
- memstate_recall: semantic search across all memories
- memstate_browse: browse the knowledge tree by keypath prefix

Requirements:
    pip install langchain-memstate langchain-openai langgraph python-dotenv

Environment variables (.env file):
    MEMSTATE_API_KEY=mst_...
    OPENAI_API_KEY=sk-...
"""
import os
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_memstate import MemstateStore, get_memstate_tools
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

API_KEY = os.getenv("MEMSTATE_API_KEY")
# Use a fixed project ID so memory persists across runs
PROJECT_ID = "demo_personal_assistant"

# Initialize the MemstateStore
# This implements the LangGraph BaseStore interface
store = MemstateStore(api_key=API_KEY, project_id=PROJECT_ID)

# Get the Memstate tools for the agent
# These allow the agent to explicitly read/write memory:
#   - memstate_remember: auto-extract facts from any text (async, uses LLM)
#   - memstate_store: store a specific value at an exact keypath (synchronous)
#   - memstate_recall: semantic search across all memories
#   - memstate_browse: browse the knowledge tree by keypath prefix
#   - memstate_get_history: get the full version history for a keypath
#   - memstate_time_travel: retrieve memory state at a specific past revision
tools = get_memstate_tools(api_key=API_KEY, project_id=PROJECT_ID)

# Create the agent
# The `prompt` parameter sets the system message
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(
    llm,
    tools=tools,
    store=store,
    prompt=(
        "You are a helpful personal assistant with access to a persistent memory system. "
        "When a user shares specific facts about themselves (name, role, preferences), "
        "use the memstate_store tool to store each fact at a precise keypath "
        "(e.g., 'users.alice.role', 'users.alice.preferences.language'). "
        "When answering questions, first use memstate_recall or memstate_browse to check what you know. "
        "Always be transparent about what you remember."
    ),
)


def run_session(session_id: str, user_input: str):
    """Run a single conversation session with the agent."""
    print(f"\n--- Session {session_id} ---")
    print(f"User: {user_input}")

    # Each session uses a unique thread_id for LangGraph checkpointing
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )

    # Print the agent's final response
    print(f"Agent: {result['messages'][-1].content}")


if __name__ == "__main__":
    print("Starting Personal Assistant Demo...")
    print(f"Project ID: {PROJECT_ID}")

    # Session 1: User shares preferences
    # The agent will call memstate_store to store these facts at precise keypaths:
    #   users.alice.role = "Senior Backend Engineer"
    #   users.alice.preferences.language = "Python"
    #   users.alice.preferences.framework = "FastAPI"
    run_session(
        "session_1",
        "Hi, I'm Alice. I'm a senior backend engineer. "
        "I prefer Python and FastAPI for new projects.",
    )

    # Session 2: User asks a question relying on memory
    # The agent will call memstate_recall to find Alice's preferences
    run_session(
        "session_2",
        "I need to build a new microservice for user authentication. "
        "What tech stack should I use?",
    )

    # Session 3: Updating memory
    # The agent will call memstate_store with the new preference.
    # Memstate creates a NEW VERSION of the language keypath — the old value is preserved.
    run_session(
        "session_3",
        "Actually, I've been learning Go lately. "
        "Let's switch my preferred language to Go, but keep my role the same.",
    )

    # Session 4: Verifying the update
    run_session(
        "session_4",
        "Remind me, what is my preferred programming language now?",
    )

    print("\n--- Current Memory Tree (users.alice) ---")
    # Inspect the memory tree directly using the store
    tree = store.browse(("users", "alice"))
    for keypath, summary in tree.items():
        print(f"  {keypath}: {summary}")
