# Multi-Agent Supervisor-Worker Architecture (LangGraph)

A self-contained, portable **Supervisor-Worker** multi-agent setup built with `langgraph` and `langgraph-supervisor`. You can move this entire folder (`langgraph-multi-agent`) anywhere on your system.

## Architecture Overview

```
                     +----------------------+
                     |      User Query      |
                     +----------+-----------+
                                |
                                v
               +--------------------------------+
               |        Supervisor Agent        |
               |  (Public LLM: e.g., GPT-4o)    |
               |   * No raw execution tools *   |
               +-------+----------------+-------+
                       |                |
         Route Research|                |Route Coding & QA
                       v                v
        +----------------------+  +----------------------+
        |   researcher_agent   |  |     coder_agent      |
        | (Local LLM: Ollama)  |  | (Local LLM: Ollama)  |
        | Tool:                |  | Tool:                |
        | search_technical_docs|  | run_code_test_suite  |
        +----------+-----------+  +----------+-----------+
                   |                         |
                   +------------+------------+
                                |
                                v
               +--------------------------------+
               |     Shared WorkflowState       |
               | - messages: list[BaseMessage]  |
               | - research_notes: str          |
               | - is_code_tested: bool         |
               +--------------------------------+
```

## Project Structure

- `src/state.py` — Custom `WorkflowState` extending `AgentState` (`messages`, `research_notes`, `is_code_tested`)
- `src/llms.py` — Step 2: Independent LLM initializers for the Supervisor (OpenAI API) and Workers (Local Ollama / OpenAI-compatible endpoint)
- `src/tools.py` — Step 3a: Scoped tools (`search_technical_docs` and `run_code_test_suite`) that update custom state fields via LangGraph `Command`
- `src/agents.py` — Step 3b: Worker agents (`researcher_agent` and `coder_agent`) built with `create_react_agent`
- `src/workflow.py` — Step 4: Supervisor graph orchestration built with `langgraph_supervisor.create_supervisor`
- `main.py` — Step 5: Compiles and executes the workflow (includes `--demo` mode for offline verification)

## Quick Start

### 1. Test the Graph Immediately (Offline Demo Mode)
Verify the graph routing, tool scoping, and custom state transitions without needing live API keys:

```bash
./.venv/bin/python main.py --demo
```

### 2. Run with Live LLMs (OpenAI + Ollama)
1. Copy `.env.example` to `.env` and fill in your API key / local model settings:
   ```bash
   cp .env.example .env
   ```
2. Ensure Ollama (or LM Studio) is running locally (e.g., `ollama pull llama3.1 && ollama serve`).
3. Run the workflow:
   ```bash
   ./.venv/bin/python main.py --query "Research our retry policy guidelines and implement/test retry_handler.py"
   ```

> **Note on Moving the Folder:** If you move `langgraph-multi-agent` to a new location, recreate the virtual environment with:
> ```bash
> rm -rf .venv && python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
> ```
