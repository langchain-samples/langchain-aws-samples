"""Deployable Deep Agent graph for LangSmith Deployment.

LangSmith Deployment imports the module-level `graph` variable from `langgraph.json`.
This exports the same support-agent shape the notebook builds: Bedrock model,
Bedrock Knowledge Base lookup, researcher delegation, S3-backed `/memories/`, and
the support reply rules from the workshop files.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env", override=True)

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain_aws import ChatBedrockConverse

from tools import query_product_kb


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _customer_namespace(rt) -> tuple[str, str, str]:
    context = getattr(rt, "context", None) or {}
    if isinstance(context, dict):
        customer_id = context.get("customer_id")
    else:
        customer_id = getattr(context, "customer_id", None)
    return ("customers", str(customer_id or "anonymous"), "memories")


def build_graph():
    """Build the deployable support agent.

    LangGraph API supplies the S3-backed store from `langgraph.json`. Keeping the
    store at the server layer avoids a custom top-level graph store, which the
    deployment runtime ignores.
    """
    model = ChatBedrockConverse(
        model=os.environ.get("AGENT_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )

    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(namespace=_customer_namespace)},
    )

    researcher = {
        "name": "researcher",
        "description": "Looks up product engineering issues in the Bedrock Knowledge Base and returns cited findings.",
        "system_prompt": (
            "You are a product engineering researcher. Use query_product_kb for every product "
            "claim, cite the documented fix exactly, save notes under /research/, and return "
            "a concise summary with source citations."
        ),
        "tools": [query_product_kb],
    }

    agents_md = _read_text(HERE / "AGENTS.md")
    skill_md = _read_text(HERE / "skills" / "support-reply" / "SKILL.md")
    system_prompt = (
        f"{agents_md}\n\n"
        "The support-reply skill is available as always-on deployment guidance:\n\n"
        f"{skill_md}\n\n"
        "Use write_todos for multi-step tickets, delegate product lookups to the "
        "researcher sub-agent, and save durable customer facts under /memories/. "
        "Expect callers to pass context={\"customer_id\": \"...\"} so memories stay "
        "isolated per customer. Then "
        "save the final answer to /final_report.md."
    )

    return create_deep_agent(
        model=model,
        tools=[],
        subagents=[researcher],
        system_prompt=system_prompt,
        backend=backend,
    )


graph = build_graph()
