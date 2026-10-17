"""Step 5: Compile and Execute the Multi-Agent Supervisor Graph.

Usage:
  # Run with live LLMs configured in .env (OpenAI + Ollama):
  python main.py

  # Run an end-to-end offline simulation to verify graph routing & state updates:
  python main.py --demo
"""

import argparse
from typing import Any, Sequence
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.workflow import create_multi_agent_graph


class DemoScriptedChatModel(BaseChatModel):
    """Deterministic tool-calling ChatModel to verify Supervisor-Worker routing offline."""

    role: str = "supervisor"

    @property
    def _llm_type(self) -> str:
        return f"demo-{self.role}"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.role == "supervisor":
            has_researched = any(
                getattr(m, "name", "") == "transfer_back_from_researcher_agent"
                or "Docs Search Result" in str(m.content)
                for m in messages
            )
            has_tested = any(
                getattr(m, "name", "") == "transfer_back_from_coder_agent"
                or "Executed pytest suite" in str(m.content)
                for m in messages
            )
            if not has_researched:
                ai_msg = AIMessage(
                    content="Routing to researcher_agent to look up retry policy documentation.",
                    tool_calls=[
                        {
                            "name": "transfer_to_researcher_agent",
                            "args": {},
                            "id": "call_sup_1",
                        }
                    ],
                )
            elif not has_tested:
                ai_msg = AIMessage(
                    content="Research complete. Routing to coder_agent to implement and run unit tests.",
                    tool_calls=[
                        {
                            "name": "transfer_to_coder_agent",
                            "args": {},
                            "id": "call_sup_2",
                        }
                    ],
                )
            else:
                ai_msg = AIMessage(
                    content=(
                        "All tasks completed! The retry policy was researched (exponential backoff "
                        "with jitter) and verified by the coder_agent's unit test suite."
                    )
                )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # Worker behavior
        last_msg = messages[-1]
        if isinstance(last_msg, ToolMessage) and getattr(last_msg, "name", "") in (
            "search_technical_docs",
            "run_code_test_suite",
        ):
            ai_msg = AIMessage(content=f"Completed tool execution: {last_msg.content}")
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        system_text = str(messages[0].content) if messages else ""
        if "Technical Researcher" in system_text:
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_technical_docs",
                        "args": {"query": "retry policy guidelines"},
                        "id": "call_res_1",
                    }
                ],
            )
        else:
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_code_test_suite",
                        "args": {"module_name": "retry_handler.py"},
                        "id": "call_cod_1",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph Supervisor-Worker workflow.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with deterministic mock LLMs to verify graph routing and custom state offline.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Research our retry policy guidelines and implement/test `retry_handler.py`.",
        help="User prompt to send to the Supervisor graph.",
    )
    args = parser.parse_args()

    if args.demo:
        print("=== Running in Offline Demo Mode (Supervisor -> Researcher -> Coder -> Supervisor) ===\n")
        app = create_multi_agent_graph(
            supervisor_llm=DemoScriptedChatModel(role="supervisor"),
            worker_llm=DemoScriptedChatModel(role="worker"),
        )
    else:
        print("=== Running with Configured LLMs (.env) ===\n")
        app = create_multi_agent_graph()

    initial_state = {
        "messages": [HumanMessage(content=args.query)],
        "research_notes": "",
        "is_code_tested": False,
    }

    final_state = app.invoke(initial_state)

    print("--- Conversation Trace ---")
    for msg in final_state["messages"]:
        sender = getattr(msg, "name", None) or msg.__class__.__name__
        if msg.content:
            print(f"[{sender}]: {msg.content}")
        if getattr(msg, "tool_calls", None):
            calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
            print(f"[{sender} -> ToolCall]: {calls}")

    print("\n--- Final Custom State ---")
    print(f"research_notes : {final_state.get('research_notes')}")
    print(f"is_code_tested : {final_state.get('is_code_tested')}")


if __name__ == "__main__":
    main()
