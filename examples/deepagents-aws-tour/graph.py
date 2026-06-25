"""Deployable Deep Agent graph for LangSmith Deployment.

LangSmith Deployment imports the module-level `graph` variable from `langgraph.json`.
This exports the same support-agent shape the notebook builds: Bedrock model,
Gateway/MCP order and ticket tools, Bedrock Knowledge Base lookup, AgentCore
Browser research, HITL-gated refunds, S3-backed `/memories/`, and the support
reply rules from the workshop files.
"""
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env", override=True)

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain_aws import ChatBedrockConverse

from mcp_client import get_gateway_tools
from tools import fetch_url, query_product_kb


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _customer_namespace(rt) -> tuple[str, str, str]:
    context = getattr(rt, "context", None) or {}
    if isinstance(context, dict):
        customer_id = context.get("customer_id")
    else:
        customer_id = getattr(context, "customer_id", None)
    return ("customers", str(customer_id or "anonymous"), "memories")


async def build_graph():
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

    gateway_tools = await get_gateway_tools()
    by_name = {t.name: t for t in gateway_tools}

    investigator = {
        "name": "investigator",
        "description": (
            "Looks up the order and customer ticket history through AgentCore Gateway, "
            "then checks known product issues in the Bedrock Knowledge Base."
        ),
        "system_prompt": (
            "You are a support investigator. For each ticket, look up the order, pull the "
            "customer's ticket history, query the product knowledge base for known issues, "
            "and save a concise findings note under /research/. Include source citations."
        ),
        "tools": [by_name["lookup_order"], by_name["lookup_customer_tickets"], query_product_kb],
    }

    browser_researcher = {
        "name": "browser_researcher",
        "description": "Reads public support docs through AgentCore Browser when a ticket includes a URL.",
        "system_prompt": (
            "You are a public-doc researcher. Use fetch_url to read URLs. Summarize only "
            "what the page actually says, include the URL, and say explicitly if it does "
            "not contain relevant support guidance."
        ),
        "tools": [fetch_url],
    }

    agents_md = _read_text(HERE / "AGENTS.md")
    skill_md = _read_text(HERE / "skills" / "support-reply" / "SKILL.md")
    system_prompt = (
        f"{agents_md}\n\n"
        "The support-reply skill is available as always-on deployment guidance:\n\n"
        f"{skill_md}\n\n"
        "Use write_todos for multi-step tickets, delegate product lookups to the "
        "investigator sub-agent, use browser_researcher for public URLs, and save "
        "durable customer facts under /memories/. "
        "Expect callers to pass context={\"customer_id\": \"...\"} so memories stay "
        "isolated per customer. If the customer explicitly asks for a refund and the "
        "facts warrant it, call issue_refund; human approval will happen before the "
        "Gateway invokes the Lambda. Then save the final answer to /final_report.md."
    )

    return create_deep_agent(
        model=model,
        tools=[*gateway_tools, query_product_kb, fetch_url],
        subagents=[investigator, browser_researcher],
        system_prompt=system_prompt,
        backend=backend,
        interrupt_on={"issue_refund": True},
    )


graph = asyncio.run(build_graph())
