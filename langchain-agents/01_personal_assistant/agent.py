import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_memstate import MemstateStore, get_memstate_tools
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# Initialize the MemstateStore
# This provides the base store for LangGraph
store = MemstateStore(
    api_key=os.getenv("MEMSTATE_API_KEY"), 
    project_id="demo_personal_assistant"
)

# Get the Memstate tools for the agent
# These allow the agent to explicitly read/write memory
tools = get_memstate_tools(
    api_key=os.getenv("MEMSTATE_API_KEY"), 
    project_id="demo_personal_assistant"
)

# Create the agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(llm, tools=tools, store=store)

def run_session(session_id: str, user_input: str):
    print(f"\n--- Session {session_id} ---")
    print(f"User: {user_input}")
    
    # Run the agent
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    )
    
    # Print the agent's response
    print(f"Agent: {result['messages'][-1].content}")

if __name__ == "__main__":
    print("Starting Personal Assistant Demo...")
    
    # Session 1: User shares preferences
    run_session(
        "session_1", 
        "Hi, I'm Alice. I'm a senior backend engineer. I prefer Python and FastAPI for new projects."
    )
    
    # Session 2: User asks a question relying on memory
    run_session(
        "session_2", 
        "I need to build a new microservice for user authentication. What tech stack should I use?"
    )
    
    # Session 3: Updating memory
    run_session(
        "session_3", 
        "Actually, I've been learning Go lately. Let's switch my preferred language to Go, but keep the role the same."
    )
    
    # Session 4: Verifying the update
    run_session(
        "session_4", 
        "Remind me, what is my preferred programming language now?"
    )
    
    print("\n--- Memory Tree ---")
    # We can inspect the memory tree directly
    tree = store.browse(("users", "alice"))
    for keypath, summary in tree.items():
        print(f"{keypath}: {summary}")
