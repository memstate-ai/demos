# Demo 2: Multi-Agent Research Pipeline

This demo shows how to use Memstate as a shared knowledge graph for a swarm of specialized LangGraph agents.

## What it demonstrates

1. **Shared Memory**: Multiple agents (Researcher, Analyst, Writer) all connect to the same Memstate `project_id`.
2. **Specialized Tools**:
   - The Researcher uses `memstate_remember` to store facts.
   - The Analyst uses `memstate_recall` (semantic search) to read facts and stores conclusions.
   - The Writer uses `memstate_browse` to read the entire tree and write a report.
3. **Conflict Resolution**: When agents write to the same or related keypaths, Memstate automatically versions the changes.

## Running the demo

From the `langchain-agents` directory:

```bash
# Make sure your .env file is set up
python 02_multi_agent_research/pipeline.py
```
