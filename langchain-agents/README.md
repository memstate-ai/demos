# LangChain & LangGraph Agents with Memstate Memory

This directory contains demos showing how to use the `langchain-memstate` package to give your agents structured, versioned memory.

## Demos

1. **[01_personal_assistant](01_personal_assistant/)**: A simple ReAct agent that remembers user preferences across multiple sessions.
2. **[02_multi_agent_research](02_multi_agent_research/)**: A LangGraph swarm of specialized agents (Researcher, Analyst, Writer) that share a single Memstate project to collaborate on a task.
3. **[03_versioned_knowledge_base](03_versioned_knowledge_base/)**: A demonstration of Memstate's version control and time-travel capabilities for auditing agent memory.

## Prerequisites

1. Python 3.10+
2. A Memstate API key (get one free at [memstate.ai/dashboard](https://memstate.ai/dashboard))
3. An OpenAI API key (for the LLM)

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your keys
   ```

4. Run the demos!
