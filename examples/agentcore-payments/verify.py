"""Offline verification for the AgentCore Payments example.

This script does not read .env, call AWS, create payment sessions, invoke a
model, or access a paid endpoint.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import botocore.session
import nbformat
from bedrock_agentcore.payments.integrations.langgraph import (
    AgentCorePaymentsConfig,
    AgentCorePaymentsMiddleware,
)
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

from build_notebook import NOTEBOOK_PATH, build_notebook
from build_setup_notebook import (
    NOTEBOOK_PATH as SETUP_NOTEBOOK_PATH,
)
from build_setup_notebook import build_notebook as build_setup_notebook
from setup_utils import (
    MANAGEMENT_ROLE,
    PAYMENT_ROLE_DEFINITIONS,
    write_env,
)


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

    setup_expected = build_setup_notebook()
    setup_actual = SETUP_NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert setup_actual == setup_expected, (
        "Run python build_setup_notebook.py."
    )

    for notebook_source in (actual, setup_actual):
        notebook = nbformat.reads(notebook_source, as_version=4)
        nbformat.validate(notebook)
        assert all(cell.get("id") for cell in notebook.cells)
        assert all(
            not cell.get("outputs")
            for cell in notebook.cells
            if cell.cell_type == "code"
        )

    assert len(PAYMENT_ROLE_DEFINITIONS) == 4
    assert PAYMENT_ROLE_DEFINITIONS[MANAGEMENT_ROLE]["deny"] == [
        "bedrock-agentcore:ProcessPayment"
    ]

    service_session = botocore.session.get_session()
    control_model = service_session.get_service_model(
        "bedrock-agentcore-control"
    )
    data_model = service_session.get_service_model("bedrock-agentcore")
    operation_inputs = {
        (control_model, "CreatePaymentCredentialProvider"): {
            "name",
            "credentialProviderVendor",
            "providerConfigurationInput",
        },
        (control_model, "CreatePaymentManager"): {
            "name",
            "authorizerType",
            "roleArn",
            "clientToken",
        },
        (control_model, "CreatePaymentConnector"): {
            "paymentManagerId",
            "credentialProviderConfigurations",
            "type",
            "clientToken",
        },
        (data_model, "CreatePaymentInstrument"): {
            "paymentManagerArn",
            "paymentConnectorId",
            "paymentInstrumentDetails",
            "paymentInstrumentType",
            "userId",
        },
        (data_model, "GetPaymentInstrumentBalance"): {
            "paymentManagerArn",
            "paymentConnectorId",
            "paymentInstrumentId",
            "userId",
            "chain",
            "token",
        },
    }
    for (service_model, operation), expected_inputs in (
        operation_inputs.items()
    ):
        actual_inputs = set(
            service_model.operation_model(operation).input_shape.members
        )
        assert expected_inputs <= actual_inputs

    with TemporaryDirectory() as temp_dir:
        test_config = Path(temp_dir) / "settings.test"
        write_env({"SAFE_TEST_KEY": "first"}, test_config)
        write_env(
            {
                "SAFE_TEST_KEY": "second",
                "SECOND_SAFE_TEST_KEY": "present",
            },
            test_config,
        )
        config_text = test_config.read_text(encoding="utf-8")
        config_lines = config_text.splitlines()
        assert sum(
            line.startswith("SAFE_TEST_KEY=") for line in config_lines
        ) == 1
        assert "SAFE_TEST_KEY=second" in config_text
        assert "SECOND_SAFE_TEST_KEY=present" in config_text

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
        "Verified both notebooks, setup configuration and role definitions, "
        "payment tool registration, and create_agent composition."
    )


if __name__ == "__main__":
    main()
