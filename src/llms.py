"""Step 2: Initialize LLM instances for Supervisor and Workers."""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI

load_dotenv()


def get_supervisor_llm() -> ChatOpenAI | AzureChatOpenAI:
    """Initialize the Supervisor LLM (Local Ollama by default, or Azure OpenAI if configured)."""
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("SUPERVISOR_MODEL", "gpt-4.1")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

    if azure_endpoint and azure_key:
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            azure_deployment=azure_deployment,
            api_version=azure_api_version,
            temperature=0.0,
        )

    model = os.getenv("SUPERVISOR_MODEL", "qwen2.5:3b")
    api_key = os.getenv("SUPERVISOR_API_KEY", "ollama")
    base_url = os.getenv("SUPERVISOR_BASE_URL", "http://localhost:11434/v1")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
    )


def get_worker_llm() -> ChatOpenAI:
    """Initialize the Worker LLM (Local Ollama by default)."""
    model = os.getenv("WORKER_MODEL", "qwen2.5:3b")
    api_key = os.getenv("WORKER_API_KEY", "ollama")
    base_url = os.getenv("WORKER_BASE_URL", "http://localhost:11434/v1")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
    )
