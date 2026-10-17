# Step-by-Step DIY Guide: Building the LangGraph Multi-Agent System from Scratch

This guide captures the exact **terminal commands**, **Python code**, and **local-model optimizations** required to build [langgraph-multi-agent](file:///home/s/code/langgraph-multi-agent) from scratch.

---

## Part 1: Terminal Actions

```bash
# 1. Pull a tool-capable local model in Ollama (3B is the sweet spot for 2C/4T CPU)
ollama pull qwen2.5:3b

# 2. Create an isolated project folder & Python virtual environment
mkdir langgraph-multi-agent && cd langgraph-multi-agent
python3 -m venv .venv
source .venv/bin/activate

# 3. Install LangGraph, the Supervisor prebuilt library, and the OpenAI/Ollama connector
pip install langgraph langgraph-supervisor langchain-openai langchain-core python-dotenv
```

---

## Part 2: The 5 Core Code Building Blocks

### 1. Define Shared State ([`src/state.py`](file:///home/s/code/langgraph-multi-agent/src/state.py))
Instead of only passing chat messages around, subclass `AgentState` to hold custom workflow variables:
```python
from typing import NotRequired
from langgraph.prebuilt.chat_agent_executor import AgentState

class WorkflowState(AgentState):
    research_notes: NotRequired[str]
    is_code_tested: NotRequired[bool]
```

### 2. Point LangChain at Local Ollama ([`src/llms.py`](file:///home/s/code/langgraph-multi-agent/src/llms.py))
Because Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1`, initialize `ChatOpenAI` instances for the Supervisor and Workers:
```python
from langchain_openai import ChatOpenAI

supervisor_llm = ChatOpenAI(
    model="qwen2.5:3b",
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    temperature=0.0,
)

worker_llm = ChatOpenAI(
    model="qwen2.5:3b",
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    temperature=0.0,
)
```

### 3. Write Scoped Tools Returning `Command` ([`src/tools.py`](file:///home/s/code/langgraph-multi-agent/src/tools.py))
To allow a tool to update custom fields in [`WorkflowState`](file:///home/s/code/langgraph-multi-agent/src/state.py#L7-L19) (`research_notes` and `is_code_tested`), return a LangGraph `Command(update={...})`:
```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command

@tool
def search_technical_docs(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Search technical documentation and update research_notes in state."""
    result = "Use exponential backoff with jitter (initial_interval=0.5s, backoff_factor=2.0, max_attempts=3)."
    return Command(
        update={
            "research_notes": result,
            "messages": [
                ToolMessage(
                    content=f"[Docs Search Result]: {result}",
                    name="search_technical_docs",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )

@tool
def run_code_test_suite(
    module_name: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Run unit tests and set is_code_tested=True in state."""
    report = f"Executed pytest suite for `{module_name}`: 4 passed, 0 failed."
    return Command(
        update={
            "is_code_tested": True,
            "messages": [
                ToolMessage(
                    content=report,
                    name="run_code_test_suite",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
```

### 4. Create the Worker Agents with `create_react_agent` ([`src/agents.py`](file:///home/s/code/langgraph-multi-agent/src/agents.py))
Bind **only** the relevant tool to each worker (`tools=[...]`) and give each worker a unique `name`:
```python
from langgraph.prebuilt import create_react_agent

researcher_agent = create_react_agent(
    model=worker_llm,
    tools=[search_technical_docs],       # <-- Only gets the search tool
    name="researcher_agent",
    state_schema=WorkflowState,
    prompt=lambda state: _filter_handoff_messages(
        state,
        "You are a Technical Researcher. Call `search_technical_docs` and summarize findings.",
    ),
)

coder_agent = create_react_agent(
    model=worker_llm,
    tools=[run_code_test_suite],         # <-- Only gets the test-runner tool
    name="coder_agent",
    state_schema=WorkflowState,
    prompt=lambda state: _filter_handoff_messages(
        state,
        "You are a Software Engineer & QA agent. Call `run_code_test_suite` for `retry_handler.py`.",
    ),
)
```

### 5. Wire the Supervisor Graph & Run ([`src/workflow.py`](file:///home/s/code/langgraph-multi-agent/src/workflow.py) & [`main.py`](file:///home/s/code/langgraph-multi-agent/main.py))
`create_supervisor` automatically generates `transfer_to_researcher_agent` and `transfer_to_coder_agent` handoff tools and compiles the state machine:
```python
from langchain_core.messages import HumanMessage
from langgraph_supervisor import create_supervisor

workflow = create_supervisor(
    agents=[researcher_agent, coder_agent],
    model=supervisor_llm,
    prompt=_build_supervisor_prompt,
    state_schema=WorkflowState,
    output_mode="full_history",
    include_agent_name="inline",
)

app = workflow.compile()
final_state = app.invoke({
    "messages": [HumanMessage(content="Research our retry policy guidelines and implement/test `retry_handler.py`.")],
    "research_notes": "",
    "is_code_tested": False,
})
```

---

## Part 3: The 2 Key Optimizations for Small Local Models (`3B`)

> [!IMPORTANT]
> Cloud frontier models (like `gpt-4.1`) handle raw multi-agent histories effortlessly, but small local models (`qwen2.5:3b`) need two lightweight prompt helpers to avoid confusion:

### 1. Filtering Supervisor Handoff Plumbing ([`_filter_handoff_messages`](file:///home/s/code/langgraph-multi-agent/src/agents.py#L11-L30))
When the Supervisor transfers control, it appends a `ToolMessage("Successfully transferred to researcher_agent")`. Small 1.5B–3B models see that trailing `ToolMessage` and assume *they already executed a tool*, causing them to skip calling `search_technical_docs`. Filtering out `transfer_*` messages from the worker's view ensures the worker calls its actual tool:
```python
def _filter_handoff_messages(state: WorkflowState, system_instruction: str) -> list[BaseMessage]:
    filtered: list[BaseMessage] = [SystemMessage(content=system_instruction)]
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and (msg.name or "").startswith("transfer_"):
            continue
        if isinstance(msg, AIMessage) and any(
            tc.get("name", "").startswith("transfer_")
            for tc in getattr(msg, "tool_calls", [])
        ):
            continue
        filtered.append(msg)
    return filtered
```

### 2. State-Aware Supervisor Prompt ([`_build_supervisor_prompt`](file:///home/s/code/langgraph-multi-agent/src/workflow.py#L13-L43))
Instead of relying on a 3B model to infer state from a long chat transcript, read `state["research_notes"]` and `state["is_code_tested"]` in Python and inject an explicit `CURRENT STATUS` instruction into the Supervisor's `SystemMessage` on every turn.
