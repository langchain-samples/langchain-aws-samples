"""MCP client for AgentCore Gateway tools.

The Gateway exposes the order and issue-management Lambdas as MCP tools. This
module discovers those tools at agent construction time and returns LangChain
tool objects that can be passed directly to create_deep_agent().
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

import httpx


class OAuthTokenAuth(httpx.Auth):
    """httpx.Auth that fetches and refreshes Cognito client-credentials tokens."""

    def __init__(self, token_url: str, client_id: str, client_secret: str):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self._token: str | None = None
        self._expires_at: datetime = datetime.min

    def _refresh(self) -> None:
        resp = httpx.post(
            self.token_url,
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._expires_at = datetime.now() + timedelta(seconds=int(data.get("expires_in", 3600)) - 60)

    def auth_header(self) -> dict[str, str]:
        if self._token is None or datetime.now() >= self._expires_at:
            self._refresh()
        return {"Authorization": f"Bearer {self._token}"}

    def auth_flow(self, request):
        request.headers.update(self.auth_header())
        yield request


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Run scripts/register_gateway.py --write-env .env "
            "after CDK deploy."
        )
    return value


def _remap_tool_names(tools: list[Any]) -> list[Any]:
    mapping = {
        "orders___lookup_order": "lookup_order",
        "orders___issue_refund": "issue_refund",
        "issues___lookup_customer_tickets": "lookup_customer_tickets",
    }
    for t in tools:
        if getattr(t, "name", None) in mapping:
            t.name = mapping[t.name]
    return tools


def _gateway_config(transport: str, auth: OAuthTokenAuth) -> dict[str, Any]:
    return {
        "transport": transport,
        "url": _required_env("GATEWAY_URL"),
        "auth": auth,
    }


async def get_gateway_tools() -> list[Any]:
    """Discover Gateway MCP tools and return them with attendee-friendly names."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    auth = OAuthTokenAuth(
        token_url=_required_env("COGNITO_TOKEN_URL"),
        client_id=_required_env("COGNITO_CLIENT_ID"),
        client_secret=_required_env("COGNITO_CLIENT_SECRET"),
    )
    preferred = os.environ.get("MCP_TRANSPORT", "streamable_http")
    transports = [preferred]
    if preferred != "http":
        transports.append("http")

    last_err: Exception | None = None
    for transport in transports:
        try:
            client = MultiServerMCPClient({"gateway": _gateway_config(transport, auth)})
            tools = await client.get_tools()
            return _remap_tool_names(tools)
        except Exception as exc:
            last_err = exc
            if transport == transports[-1]:
                break

    raise RuntimeError(f"Could not discover Gateway MCP tools: {last_err}") from last_err


async def print_gateway_tools() -> list[str]:
    """Convenience helper for notebook smoke tests."""
    tools = await get_gateway_tools()
    names = sorted(t.name for t in tools)
    print("Gateway tools:", ", ".join(names))
    return names
