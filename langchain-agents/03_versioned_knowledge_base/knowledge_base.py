"""
Demo 3: Versioned Knowledge Base with Time-Travel
==================================================
Demonstrates Memstate's version control and time-travel capabilities.

The `memstate_store` tool writes a fact directly to a specific keypath.
Every subsequent write to the same keypath creates a new version — the old
value is NEVER deleted. This means you can:
  - See the full audit trail of how a fact evolved
  - Reconstruct the exact state of memory at any past revision
  - Debug agent decisions by replaying the memory state from that moment

Key concepts:
- memstate_store tool: synchronous, direct keypath write (creates new version)
- store.get_history(): returns the full audit trail for a keypath
- store.get_at_revision(): time-travel to any past revision
- store.put(): async ingestion (uses LLM extraction, best for natural language)

Requirements:
    pip install langchain-memstate python-dotenv

Environment variables (.env file):
    MEMSTATE_API_KEY=mst_...
"""
import os
import uuid
from dotenv import load_dotenv
from langchain_memstate import MemstateStore, get_memstate_tools

# Load environment variables
load_dotenv()

API_KEY = os.getenv("MEMSTATE_API_KEY")
# Use a unique project ID per run so demos don't bleed into each other
PROJECT_ID = f"demo_versioned_kb_{uuid.uuid4().hex[:8]}"


def run_demo():
    print("Starting Versioned Knowledge Base Demo...")
    print(f"Project ID: {PROJECT_ID}\n")

    store = MemstateStore(api_key=API_KEY, project_id=PROJECT_ID)
    tools = get_memstate_tools(api_key=API_KEY, project_id=PROJECT_ID)
    tool_map = {t.name: t for t in tools}

    # --- Simulate a project evolving over time ---
    # We use the `memstate_store` tool for direct, synchronous keypath writes.
    # Each write to an existing keypath creates a new version automatically.

    print("--- Time t=0: Initial Architecture ---")
    r = tool_map["memstate_store"].invoke({
        "keypath": "project.backend.auth.provider",
        "content": "Basic Auth",
    })
    print(f"  {r}")

    tool_map["memstate_store"].invoke({
        "keypath": "project.backend.database.engine",
        "content": "MySQL 5.7",
    })
    print("  Stored: database.engine = 'MySQL 5.7'")

    print("\n--- Time t=1: Security Upgrade ---")
    # Calling memstate_store on an existing keypath creates a NEW VERSION.
    # The old value ('Basic Auth') is preserved in history.
    r = tool_map["memstate_store"].invoke({
        "keypath": "project.backend.auth.provider",
        "content": "OAuth2 + JWT",
    })
    print(f"  {r}")

    print("\n--- Time t=2: Database Migration ---")
    tool_map["memstate_store"].invoke({
        "keypath": "project.backend.database.engine",
        "content": "PostgreSQL 16",
    })
    print("  Updated: database.engine = 'PostgreSQL 16'  (v1 'MySQL 5.7' preserved)")

    print("\n--- Time t=3: Adding MFA ---")
    r = tool_map["memstate_store"].invoke({
        "keypath": "project.backend.auth.provider",
        "content": "OAuth2 + JWT + MFA",
    })
    print(f"  {r}")

    # --- Audit Trail ---
    print("\n--- Full Audit Trail for Auth Provider ---")
    # get_history() uses the (namespace, key) format
    # keypath "project.backend.auth.provider" → namespace=("project","backend","auth"), key="provider"
    history = store.get_history(("project", "backend", "auth"), "provider")
    print(f"Found {len(history)} versions:")
    for version in history:
        ver = version.get("version", "?")
        summary = version.get("summary", version.get("content", ""))[:80]
        superseded = version.get("superseded_by")
        status = f"-> superseded by v{superseded}" if superseded else "(current)"
        print(f"  v{ver} {status}: {summary}")

    assert len(history) >= 2, f"Expected at least 2 versions, got {len(history)}"

    # --- Time-Travel ---
    # Find the earliest revision from history to time-travel to
    earliest_revision = min(v.get("version", 999) for v in history)
    print(f"\n--- Time-Travel: Reconstruct state at revision {earliest_revision} ---")
    print("(This is what the agent knew when auth was still 'Basic Auth')")
    snapshot = store.get_at_revision(
        ("project", "backend"), at_revision=earliest_revision
    )
    print(f"  State at revision {earliest_revision}:")
    for keypath, summary in snapshot.items():
        print(f"    {keypath}: {summary[:80]}")

    # --- Current State ---
    print("\n--- Current State (latest versions) ---")
    current = store.browse(("project", "backend"))
    for keypath, summary in current.items():
        print(f"  {keypath}: {summary[:80]}")

    print("\nDemo complete!")


if __name__ == "__main__":
    run_demo()
