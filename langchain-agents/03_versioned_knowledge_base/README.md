# Demo 3: Versioned Knowledge Base

This demo highlights Memstate's version control and time-travel capabilities. Unlike standard vector stores that overwrite or duplicate data, Memstate maintains a complete, immutable history of every fact.

## What it demonstrates

1. **Automatic Versioning**: When you `put()` or `remember()` a fact at an existing keypath, the old value is preserved in history.
2. **Audit Trails**: Using `store.get_history()`, you can see exactly how a fact evolved over time.
3. **Time-Travel**: Using `store.get_at_revision()`, you can reconstruct the exact state of your agent's memory at any point in the past. This is crucial for debugging ("Why did the agent make that decision yesterday?").

## Running the demo

From the `langchain-agents` directory:

```bash
# Make sure your .env file is set up
python 03_versioned_knowledge_base/knowledge_base.py
```
