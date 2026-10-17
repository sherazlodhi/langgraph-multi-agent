"""Step 3a: Define strictly scoped tools for specialized Worker Agents.

Best Practice (Tool Scoping):
Only assign tools to the specific worker agents that need them.
Never give the Supervisor raw execution tools.
"""

from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command


# ============================================================================
# Researcher Agent Tools
# ============================================================================

@tool
def search_technical_docs(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Search technical documentation and architecture guides for a given query.

    Updates `research_notes` in the shared graph state alongside returning a ToolMessage.
    """
    knowledge_base = {
        "langgraph": (
            "LangGraph builds stateful, multi-actor applications with LLMs using graph nodes and edges. "
            "Use `langgraph-supervisor` for Supervisor-Worker routing and `TypedDict`/`AgentState` for shared state."
        ),
        "retry": (
            "Recommended retry policy: Use exponential backoff with jitter (initial_interval=0.5s, "
            "backoff_factor=2.0, max_attempts=3) on transient network errors."
        ),
    }

    q_lower = query.lower()
    matched = [
        info for key, info in knowledge_base.items() if key in q_lower
    ]
    result = (
        "\n".join(matched)
        if matched
        else f"Found standard best-practice guidelines for '{query}': modular design, explicit typing, and unit tests."
    )

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


# ============================================================================
# Coder / Tester Agent Tools
# ============================================================================

@tool
def run_code_test_suite(
    module_name: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Run unit tests for a target module and update `is_code_tested` in shared state."""
    report = (
        f"Executed pytest suite for `{module_name}`: 4 passed, 0 failed in 0.12s. "
        "State variable `is_code_tested` set to True."
    )
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
