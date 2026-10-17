"""Step 4: Create the Supervisor Workflow using langgraph-supervisor."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph_supervisor import create_supervisor

from src.agents import build_researcher_agent, build_coder_agent
from src.llms import get_supervisor_llm, get_worker_llm
from src.state import WorkflowState


def _build_supervisor_prompt(state: WorkflowState) -> list[BaseMessage]:
    """Build a state-aware prompt for the Supervisor so local 3B models route accurately."""
    research_done = bool(state.get("research_notes"))
    code_tested = bool(state.get("is_code_tested"))

    if not research_done:
        next_step_hint = (
            "CURRENT STATUS: Research is NOT done yet (`research_notes` is empty).\n"
            "ACTION REQUIRED: Call `transfer_to_researcher_agent` now."
        )
    elif not code_tested:
        next_step_hint = (
            f"CURRENT STATUS: Research is complete (`research_notes`: {state.get('research_notes')}), "
            "but `is_code_tested` is False.\n"
            "ACTION REQUIRED: Call `transfer_to_coder_agent` now."
        )
    else:
        next_step_hint = (
            "CURRENT STATUS: Both research and code testing are COMPLETE (`is_code_tested` is True).\n"
            "ACTION REQUIRED: Do NOT call any transfer tools. Provide a concise final summary to the user."
        )

    system_msg = SystemMessage(
        content=(
            "You are a Supervisor managing two specialized worker agents:\n"
            "1. `researcher_agent`: Searches technical documentation and records `research_notes`.\n"
            "2. `coder_agent`: Runs the unit test suite and sets `is_code_tested` to True.\n\n"
            f"{next_step_hint}"
        )
    )
    return [system_msg] + list(state["messages"])


def create_multi_agent_graph(
    supervisor_llm: BaseChatModel | None = None,
    worker_llm: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """Build and compile the Supervisor-Worker LangGraph workflow."""
    sup_llm = supervisor_llm or get_supervisor_llm()
    wrk_llm = worker_llm or get_worker_llm()

    researcher = build_researcher_agent(wrk_llm)
    coder = build_coder_agent(wrk_llm)

    workflow = create_supervisor(
        agents=[researcher, coder],
        model=sup_llm,
        prompt=_build_supervisor_prompt,
        state_schema=WorkflowState,
        output_mode="full_history",
        include_agent_name="inline",
    )

    return workflow.compile()
