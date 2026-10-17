"""Step 3b: Create specialized Worker Agents using isolated LLM and tool configurations."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from src.state import WorkflowState
from src.tools import search_technical_docs, run_code_test_suite


def _filter_handoff_messages(
    state: WorkflowState, system_instruction: str
) -> list[BaseMessage]:
    """Filter out Supervisor handoff plumbing messages for local Worker LLMs.

    Smaller local models (like qwen2.5:1.5b or llama3) can mistake a trailing
    `ToolMessage(name='transfer_to_...')` for their own completed tool execution.
    Filtering handoff tool messages ensures the worker focuses on the user task
    and invokes its own scoped tools.
    """
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


def build_researcher_agent(worker_llm: BaseChatModel):
    """Create the Researcher worker agent scoped only to documentation search tools."""
    system_prompt = (
        "You are a specialized Technical Researcher agent.\n"
        "You MUST call the `search_technical_docs` tool to look up documentation for the user's request.\n"
        "After receiving the tool output, summarize the technical findings concisely."
    )
    return create_react_agent(
        model=worker_llm,
        tools=[search_technical_docs],
        name="researcher_agent",
        state_schema=WorkflowState,
        prompt=lambda state: _filter_handoff_messages(state, system_prompt),
    )


def build_coder_agent(worker_llm: BaseChatModel):
    """Create the Coder/Tester worker agent scoped only to code verification tools."""
    system_prompt = (
        "You are a specialized Software Engineer and QA agent.\n"
        "You MUST call the `run_code_test_suite` tool for the target module (`retry_handler.py`) "
        "to verify the implementation and set `is_code_tested` to True.\n"
        "After running the tool, report the implementation and unit test results."
    )
    return create_react_agent(
        model=worker_llm,
        tools=[run_code_test_suite],
        name="coder_agent",
        state_schema=WorkflowState,
        prompt=lambda state: _filter_handoff_messages(state, system_prompt),
    )
