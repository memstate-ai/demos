import os
from dotenv import load_dotenv
from langchain_memstate import MemstateStore

# Load environment variables
load_dotenv()

# Initialize the MemstateStore directly (no agent needed for this demo)
store = MemstateStore(
    api_key=os.getenv("MEMSTATE_API_KEY"), 
    project_id="demo_versioned_kb"
)

def run_demo():
    print("Starting Versioned Knowledge Base Demo...\n")
    
    # Simulate a project evolving over time
    print("--- Time t=0: Initial Architecture ---")
    store.put(("project", "backend", "auth"), "provider", {"value": "Basic Auth"})
    store.put(("project", "backend", "database"), "engine", {"value": "MySQL 5.7"})
    print("Stored initial architecture.")
    
    print("\n--- Time t=1: Security Upgrade ---")
    # The team migrates to OAuth2
    # Memstate automatically creates a new version; the old one is NEVER deleted
    store.put(("project", "backend", "auth"), "provider", {"value": "OAuth2 + JWT"})
    print("Upgraded auth to OAuth2 + JWT.")
    
    print("\n--- Time t=2: Database Migration ---")
    store.put(("project", "backend", "database"), "engine", {"value": "PostgreSQL 16"})
    print("Migrated database to PostgreSQL 16.")
    
    print("\n--- Time t=3: Adding MFA ---")
    store.put(("project", "backend", "auth"), "provider", {"value": "OAuth2 + JWT + MFA"})
    print("Added MFA to auth provider.")
    
    print("\n--- Full Audit Trail for Auth Provider ---")
    # See the full audit trail
    history = store.get_history(("project", "backend", "auth"), "provider")
    for version in history:
        print(f"v{version['version']} [{version.get('created_at', '')}]: {version.get('summary', '')}")
        print(f"  Value: {version.get('value')}")
        
    print("\n--- Time-Travel: What did the architecture look like at Revision 2? ---")
    # Reconstruct the exact state of memory at a past point in time
    # Note: revision numbers are global across the project
    try:
        snapshot = store.get_at_revision(("project", "backend"), at_revision=2)
        print("Project state at revision 2:")
        for keypath, summary in snapshot.items():
            print(f"  {keypath}: {summary}")
    except Exception as e:
        print(f"Could not fetch revision 2: {e}")
        
    print("\n--- Current State ---")
    current = store.browse(("project", "backend"))
    for keypath, summary in current.items():
        print(f"  {keypath}: {summary}")

if __name__ == "__main__":
    run_demo()
