"""
Demo 2: Multi-Agent Research Pipeline with Shared Memory
=========================================================
Shows how to use Memstate as a shared knowledge graph for a swarm of
specialized LangGraph agents.

Three agents share the same Memstate project_id:
  1. Researcher  — gathers facts and stores them with memstate_remember
  2. Analyst     — reads facts with memstate_recall, adds analysis
  3. Writer      — browses the full tree with memstate_browse, writes a report

Key concepts:
- All agents write to the same project_id, creating a shared knowledge graph
- Memstate automatically handles versioning if agents update the same keypath
- Agents communicate through memory, not through direct message passing
- memstate_remember: RECOMMENDED — async LLM-extracted write for any text
- memstate_store: synchronous direct keypath write for precise facts

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
# Use a unique project ID per run so demos don't bleed into each other
PROJECT_ID = f"demo_multi_agent_{uuid.uuid4().hex[:8]}"

print(f"Using shared project ID: {PROJECT_ID}")

# Initialize the shared MemstateStore
# All three agents will connect to this same project
store = MemstateStore(api_key=API_KEY, project_id=PROJECT_ID)

# Get the Memstate tools — same tools, same project, for all agents
tools = get_memstate_tools(api_key=API_KEY, project_id=PROJECT_ID)

# Create the specialized agents
# The `prompt` parameter sets each agent's system message / role
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Agent 1: The Researcher
# Gathers raw facts and stores them using memstate_remember
# Memstate's custom LLMs automatically extract and organize the facts
researcher = create_react_agent(
    llm,
    tools=tools,
    store=store,
    prompt=(
        "You are a researcher. Your job is to find facts and use the "
        "memstate_remember tool to store them in the shared memory. "
        "Memstate will automatically extract and organize the facts. "
        "Be thorough and store all key facts you discover."
    ),
)

# Agent 2: The Analyst
# Reads facts with memstate_recall, synthesizes, and stores conclusions
analyst = create_react_agent(
    llm,
    tools=tools,
    store=store,
    prompt=(
        "You are an analyst. Your job is to use memstate_recall to read the "
        "facts stored by the researcher, analyze them, and use memstate_remember "
        "to store your conclusions and insights."
    ),
)

# Agent 3: The Writer
# Reads everything with memstate_browse and produces a final report
writer = create_react_agent(
    llm,
    tools=tools,
    store=store,
    prompt=(
        "You are a technical writer. Your job is to use memstate_browse to read "
        "all the research and analysis stored in memory, then write a clear, "
        "concise final summary."
    ),
)


def run_pipeline():
    print("\nStarting Multi-Agent Research Pipeline...\n")

    # Phase 1: Researcher
    print("--- Phase 1: Researcher ---")
    research_prompt = (
        "Research the memory system architecture for Memstate AI. "
        "Key facts: "
        "1. It uses hierarchical keypaths (like users.alice.preferences.language). "
        "2. Custom LLMs are used for automatic fact extraction from natural language. "
        "3. Every write automatically creates a new version (immutable history). "
        "4. Semantic search is supported across all memories. "
        "Store all these facts in memory using memstate_remember."
    )
    print(f"Task: {research_prompt}")
    researcher.invoke({"messages": [{"role": "user", "content": research_prompt}]})
    print("Researcher finished storing facts.\n")

    # Phase 2: Analyst
    print("--- Phase 2: Analyst ---")
    analysis_prompt = (
        "Recall the facts about Memstate AI's architecture. "
        "Analyze why hierarchical keypaths are better than flat vector stores for AI agents. "
        "Consider: precision retrieval, no duplicate facts, audit trails, and token efficiency. "
        "Store your analysis in memory using memstate_remember."
    )
    print(f"Task: {analysis_prompt}")
    analyst.invoke({"messages": [{"role": "user", "content": analysis_prompt}]})
    print("Analyst finished storing conclusions.\n")

    # Phase 3: Writer
    print("--- Phase 3: Writer ---")
    writer_prompt = (
        "Browse the entire memory for this project. "
        "Write a 2-paragraph technical summary that combines the architecture facts "
        "and the analysis. Make it suitable for a developer blog post."
    )
    print(f"Task: {writer_prompt}")
    result = writer.invoke({"messages": [{"role": "user", "content": writer_prompt}]})

    print("\n--- Final Report ---")
    print(result["messages"][-1].content)

    print("\n--- Shared Memory Tree ---")
    # Inspect the shared knowledge graph that all three agents built together
    tree = store.browse(("research",))
    for keypath, summary in tree.items():
        print(f"  {keypath}: {summary[:80]}")


if __name__ == "__main__":
    run_pipeline()
