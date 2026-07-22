"""Offline verification for the AgentCore Payments example.

This script does not read .env, call AWS, create payment sessions, invoke a
model, or access a paid endpoint.
"""

from __future__ import annotations

from unittest.mock import patch

import nbformat
from bedrock_agentcore.payments.integrations.langgraph import (
    AgentCorePaymentsConfig,
    AgentCorePaymentsMiddleware,
)
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

from build_notebook import NOTEBOOK_PATH, build_notebook


class StubModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "offline-verification-stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("Offline verification must not invoke the model.")

    def bind_tools(self, tools, **kwargs):
        return self


def main() -> None:
    expected = build_notebook()
    actual = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert actual == expected, "Run python build_notebook.py."

    notebook = nbformat.reads(actual, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.get("id") for cell in notebook.cells)
    assert all(
        not cell.get("outputs")
        for cell in notebook.cells
        if cell.cell_type == "code"
    )

    config = AgentCorePaymentsConfig(
        payment_manager_arn=(
            "arn:aws:bedrock-agentcore:us-west-2:"
            "000000000000:payment-manager/test"
        ),
        user_id="offline-test-user",
        payment_instrument_id="offline-test-instrument",
        region="us-west-2",
        network_preferences_config=["eip155:84532", "base-sepolia"],
        auto_session=True,
        auto_session_budget="1.00",
        auto_session_expiry_minutes=60,
    )

    with patch(
        "bedrock_agentcore.payments.integrations.langgraph."
        "middleware.PaymentManager"
    ):
        middleware = AgentCorePaymentsMiddleware(config)
        tool_names = {tool.name for tool in middleware.tools}
        assert "http_request" in tool_names
        agent = create_agent(
            model=StubModel(),
            tools=[],
            middleware=[middleware],
        )
        assert agent is not None

    print(
        "Verified notebook generation/schema, empty outputs, payment tool "
        "registration, and create_agent middleware composition."
    )


if __name__ == "__main__":
    main()
