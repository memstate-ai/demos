import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_memstate import MemstateStore, get_memstate_tools
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# Shared Memstate Project ID for the whole team
PROJECT_ID = "demo_multi_agent"
API_KEY = os.getenv("MEMSTATE_API_KEY")

# Initialize the shared MemstateStore
store = MemstateStore(api_key=API_KEY, project_id=PROJECT_ID)

# Get the Memstate tools
tools = get_memstate_tools(api_key=API_KEY, project_id=PROJECT_ID)

# Create the specialized agents
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Agent 1: The Researcher (gathers facts and stores them)
researcher = create_react_agent(
    llm, 
    tools=tools, 
    store=store,
    state_modifier="You are a researcher. Your job is to find facts and use memstate_remember to store them in the shared memory."
)

# Agent 2: The Analyst (reads facts, analyzes, and stores conclusions)
analyst = create_react_agent(
    llm, 
    tools=tools, 
    store=store,
    state_modifier="You are an analyst. Your job is to use memstate_recall to read facts, analyze them, and use memstate_remember to store your conclusions."
)

# Agent 3: The Writer (reads everything and produces a final report)
writer = create_react_agent(
    llm, 
    tools=tools, 
    store=store,
    state_modifier="You are a technical writer. Your job is to use memstate_browse to read all the research and analysis, and write a final summary."
)

def run_pipeline():
    print("Starting Multi-Agent Research Pipeline...\n")
    
    # Step 1: Researcher
    print("--- Phase 1: Researcher ---")
    research_prompt = "Research the memory system architecture for Memstate AI. Note that it uses hierarchical keypaths and custom LLMs for extraction. Store these facts."
    print(f"Task: {research_prompt}")
    researcher.invoke({"messages": [{"role": "user", "content": research_prompt}]})
    print("Researcher finished storing facts.\n")
    
    # Step 2: Analyst
    print("--- Phase 2: Analyst ---")
    analysis_prompt = "Recall the facts about Memstate AI's architecture. Analyze why hierarchical keypaths are better than flat vector stores for AI agents. Store your analysis."
    print(f"Task: {analysis_prompt}")
    analyst.invoke({"messages": [{"role": "user", "content": analysis_prompt}]})
    print("Analyst finished storing conclusions.\n")
    
    # Step 3: Writer
    print("--- Phase 3: Writer ---")
    writer_prompt = "Browse the memory for the Memstate AI project. Write a short, 2-paragraph technical summary combining the architecture facts and the analysis."
    print(f"Task: {writer_prompt}")
    result = writer.invoke({"messages": [{"role": "user", "content": writer_prompt}]})
    
    print("\n--- Final Report ---")
    print(result['messages'][-1].content)
    
    print("\n--- Shared Memory Tree ---")
    # Inspect the shared knowledge graph
    tree = store.browse(())
    for keypath, summary in tree.items():
        print(f"{keypath}: {summary}")

if __name__ == "__main__":
    run_pipeline()
