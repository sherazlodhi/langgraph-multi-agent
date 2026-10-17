"""Shared state definition for the Supervisor-Worker multi-agent graph."""

from typing import NotRequired
from langgraph.prebuilt.chat_agent_executor import AgentState


class WorkflowState(AgentState):
    """Custom shared state extending LangGraph's prebuilt AgentState.

    Attributes:
        messages: Conversation history shared across agents (inherited from AgentState).
        remaining_steps: Managed step counter used by ReAct agents (inherited from AgentState).
        research_notes: Structured notes gathered by the Researcher worker.
        is_code_tested: Flag indicating whether code changes have been verified by tests.
    """

    research_notes: NotRequired[str]
    is_code_tested: NotRequired[bool]
