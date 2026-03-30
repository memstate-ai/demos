# Memstate AI Demos

This repository contains example code and tutorials for building memory-powered AI agents using [Memstate AI](https://memstate.ai).

## LangChain & LangGraph Agents

The `langchain-agents` directory contains three demos showing how to integrate Memstate with LangChain and LangGraph:

1. **[Personal Assistant](langchain-agents/01_personal_assistant/)**: A simple ReAct agent that remembers user preferences across multiple sessions.
2. **[Multi-Agent Research](langchain-agents/02_multi_agent_research/)**: A LangGraph swarm of specialized agents (Researcher, Analyst, Writer) that share a single Memstate project to collaborate on a task.
3. **[Versioned Knowledge Base](langchain-agents/03_versioned_knowledge_base/)**: A demonstration of Memstate's version control and time-travel capabilities for auditing agent memory.

## Getting Started

1. Get a free API key at [memstate.ai/dashboard](https://memstate.ai/dashboard).
2. Clone this repository.
3. Navigate to the specific demo directory and follow its README instructions.

## What is Memstate?

Memstate gives your AI agents structured, versioned memory they can navigate. Custom LLM models extract keypaths automatically, detect conflicts, and compress context.

- **Keypaths**: Memories are organized hierarchically using dot-separated paths like `auth.provider`.
- **Version Control**: Every memory is versioned. When content is updated, the previous version is preserved in history.
- **Token Efficiency**: Structured `keypath = value` atoms instead of text blobs.
- **Time-Travel Queries**: See memory state at any point in history.

Learn more in the [Memstate Documentation](https://memstate.ai/docs).
