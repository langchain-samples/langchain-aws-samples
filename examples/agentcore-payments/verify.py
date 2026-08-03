"""Offline verification for the AgentCore Payments example.

This script does not read .env, call AWS, create payment sessions, invoke a
model, or access a paid endpoint.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextlib import redirect_stdout
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import botocore.session
import nbformat
from bedrock_agentcore.payments.integrations.langgraph import (
    AgentCorePaymentsConfig,
    AgentCorePaymentsMiddleware,
)
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langsmith.schemas import ExampleCreate

from build_notebook import NOTEBOOK_PATH, build_notebook
from build_setup_notebook import (
    NOTEBOOK_PATH as SETUP_NOTEBOOK_PATH,
)
from build_setup_notebook import build_notebook as build_setup_notebook
from setup_utils import (
    MANAGEMENT_ROLE,
    PAYMENT_ROLE_DEFINITIONS,
    is_not_found,
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


def code_cell(notebook_source: str, marker: str) -> str:
    """Return one generated code cell identified by a stable source marker."""
    notebook = nbformat.reads(notebook_source, as_version=4)
    matches = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and marker in cell.source
    ]
    assert len(matches) == 1, f"Expected one code cell containing {marker!r}."
    return matches[0]


def notebook_source_signature(notebook_source: str) -> list[tuple]:
    """Describe reviewable cell content while ignoring saved run outputs."""
    notebook = nbformat.reads(notebook_source, as_version=4)
    return [
        (cell.cell_type, cell.get("id"), cell.source)
        for cell in notebook.cells
    ]


def verify_cleanup_workflow(setup_notebook_source: str) -> None:
    """Check cleanup guards, API order, and local-value handling offline."""
    cleanup_source = code_cell(
        setup_notebook_source,
        "EXPECTED_CLEANUP_CONFIRMATION",
    )
    expected_guard = "DELETE AGENTCORE PAYMENTS TEST RESOURCES"
    assert expected_guard in cleanup_source
    assert 'CLEANUP_CONFIRMATION = ""' in cleanup_source

    deletion_markers = [
        "list_payment_sessions",
        "delete_payment_session",
        "delete_payment_instrument",
        "delete_payment_connector",
        "delete_payment_manager",
        "delete_payment_credential_provider",
        "delete_payment_roles(cleanup_region)",
        'write_env({key: "" for key in cleanup_keys})',
    ]
    marker_positions = [
        cleanup_source.index(marker) for marker in deletion_markers
    ]
    assert marker_positions == sorted(marker_positions)
    assert "if is_not_found(error):" in cleanup_source
    assert "os.environ[key] = \"\"" in cleanup_source

    assert "EXPECTED_LANGSMITH_CONFIRMATION" not in setup_notebook_source
    assert "delete_dataset(" not in setup_notebook_source

    local_cleanup_source = code_cell(
        setup_notebook_source,
        "EXPECTED_LOCAL_CLEAR_CONFIRMATION",
    )
    assert "CLEAR LOCAL COINBASE VALUES" in local_cleanup_source
    assert 'LOCAL_CLEAR_CONFIRMATION = ""' in local_cleanup_source
    for key in (
        "COINBASE_API_KEY_ID",
        "COINBASE_API_KEY_SECRET",
        "COINBASE_WALLET_SECRET",
        "LINKED_EMAIL",
    ):
        assert key in local_cleanup_source


class FakeLangSmithClient:
    """In-memory dataset client used to prove idempotent notebook behavior."""

    def __init__(self) -> None:
        self.dataset = SimpleNamespace(
            id=UUID("00000000-0000-0000-0000-000000000123"),
            name="agentcore-payments-policy-evals",
        )
        self.dataset_exists = False
        self.examples: dict[UUID, dict] = {}
        self.projects: set[str] = set()

    def has_dataset(self, *, dataset_name: str) -> bool:
        assert dataset_name == self.dataset.name
        return self.dataset_exists

    def read_dataset(self, *, dataset_name: str):
        assert self.dataset_exists
        assert dataset_name == self.dataset.name
        return self.dataset

    def create_dataset(self, *, dataset_name: str, description: str):
        assert dataset_name == self.dataset.name
        assert description
        self.dataset_exists = True
        return self.dataset

    def create_examples(self, *, dataset_id, examples):
        assert dataset_id == self.dataset.id
        for example in examples:
            assert example["id"] not in self.examples
            self.examples[example["id"]] = example

    def list_examples(self, *, dataset_id, example_ids):
        assert dataset_id == self.dataset.id
        return iter(
            SimpleNamespace(id=example_id)
            for example_id in example_ids
            if example_id in self.examples
        )

    def update_examples(self, *, dataset_id, updates):
        assert dataset_id == self.dataset.id
        for update in updates:
            assert update["id"] in self.examples
            self.examples[update["id"]] = update

    def create_project(self, project_name: str, **kwargs):
        assert kwargs["upsert"] is True
        assert kwargs["description"]
        self.projects.add(project_name)
        return SimpleNamespace(name=project_name)

    def list_runs(self, *, project_name: str, **kwargs):
        assert project_name in self.projects
        assert kwargs["is_root"] is True
        assert kwargs["start_time"]
        return iter([SimpleNamespace(name="offline-trace")])

    def get_run_url(self, *, run, project_name: str) -> str:
        assert run.name == "offline-trace"
        assert project_name in self.projects
        return "https://aws.smith.langchain.com/o/offline/r/offline"

    def flush(self) -> None:
        return None


class StubJudgeModel:
    """Structured-output Bedrock stand-in that never invokes a model."""

    def __init__(self, **kwargs) -> None:
        assert kwargs["model"]
        assert kwargs["region_name"]

    def with_structured_output(self, schema):
        assert schema
        return self

    def invoke(self, messages):
        assert messages
        rendered = str(messages)
        assert "HTTP evidence:" in rendered
        assert "paid test data" in rendered
        assert "Observed spending (USD):" in rendered
        return {
            "reasoning": "The response accurately reports the outcome.",
            "is_correct": True,
        }


class FakePaymentConfig:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        if "payment_session_id" not in kwargs:
            self.payment_session_id = None


class FakePaymentMiddleware:
    def __init__(self, config) -> None:
        self.config = config


class FakePaymentManager:
    def __init__(self) -> None:
        self.sessions: dict[str, Decimal] = {}
        self.counter = 0

    def create_payment_session(self, *, limits, **kwargs):
        self.counter += 1
        session_id = f"explicit-{self.counter}"
        budget = Decimal(limits["maxSpendAmount"]["value"])
        self.sessions[session_id] = budget
        return {"paymentSessionId": session_id}

    def get_payment_session(self, *, payment_session_id: str, **kwargs):
        budget = self.sessions[payment_session_id]
        spent = Decimal("0") if budget <= Decimal("0.0001") else Decimal("0.01")
        return {
            "limits": {
                "maxSpendAmount": {"value": str(budget), "currency": "USD"}
            },
            "availableLimits": {
                "availableSpendAmount": {
                    "value": str(budget - spent),
                    "currency": "USD",
                }
            },
        }


def verify_langsmith_workflow(notebook_source: str) -> None:
    """Exercise dataset, target, and evaluator cells without external calls."""
    notebook = nbformat.reads(notebook_source, as_version=4)
    learning_markers = [
        "trace_payment_config, trace_session_id",
        "trace_agent = build_agent",
        "trace_result = trace_agent.invoke",
        "dataset_examples = [",
        "def make_payment_target",
        "def payment_outcome_correct",
        "baseline_results =",
        "audit_results =",
    ]
    marker_positions = [
        next(
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code" and marker in cell.source
        )
        for marker in learning_markers
    ]
    assert marker_positions == sorted(marker_positions)

    agent_source = code_cell(notebook_source, "trace_agent = build_agent")
    assert "create_agent(" in agent_source
    assert "tools=[]" in agent_source
    assert "middleware=[payment_middleware]" in agent_source
    assert "AUDIT_PROMPT = BASELINE_PROMPT +" not in agent_source
    assert "Summarize only the API response body" in agent_source
    assert "the application reports authoritative" in agent_source

    dataset_source = code_cell(notebook_source, "dataset_examples = [")
    fake_client = FakeLangSmithClient()
    dataset_namespace = {
        "DATASET_NAME": fake_client.dataset.name,
        "langsmith_client": fake_client,
        "request": "Use the paid test endpoint.",
    }
    with patch("langsmith.Client", return_value=fake_client):
        with redirect_stdout(StringIO()):
            exec(dataset_source, dataset_namespace)
            first_example_ids = set(fake_client.examples)
            exec(dataset_source, dataset_namespace)
    assert len(fake_client.examples) == 3
    assert set(fake_client.examples) == first_example_ids
    for example in fake_client.examples.values():
        ExampleCreate.model_validate(example)
    expected_outcomes = {
        example["outputs"]["expected_payment_outcome"]
        for example in fake_client.examples.values()
    }
    assert expected_outcomes == {"paid", "rejected"}

    payment_integration_source = code_cell(
        notebook_source,
        "trace_payment_config, trace_session_id",
    )
    target_source = code_cell(notebook_source, "def make_payment_target")
    fake_manager = FakePaymentManager()
    invoke_configs: list[dict] = []

    def fake_build_agent(middleware, system_prompt: str):
        assert system_prompt

        class FakeAgent:
            def invoke(self, payload, config):
                invoke_configs.append(config)
                payment_config = middleware.config
                if getattr(payment_config, "auto_session", False):
                    budget = Decimal(payment_config.auto_session_budget)
                    payment_config.payment_session_id = "automatic-1"
                    fake_manager.sessions["automatic-1"] = budget
                else:
                    budget = fake_manager.sessions[
                        payment_config.payment_session_id
                    ]
                if budget <= Decimal("0.0001"):
                    messages = [
                        SimpleNamespace(
                            type="tool",
                            name="http_request",
                            content="PAYMENT ERROR: limit exceeded",
                        ),
                        SimpleNamespace(
                            type="ai",
                            name=None,
                            content="Payment was blocked.",
                        ),
                    ]
                else:
                    messages = [
                        SimpleNamespace(
                            type="tool",
                            name="http_request",
                            content=json.dumps(
                                {
                                    "statusCode": 200,
                                    "headers": {
                                        "payment-response": "drop-this"
                                    },
                                    "body": {"result": "paid test data"},
                                }
                            ),
                        ),
                        SimpleNamespace(
                            type="ai",
                            name=None,
                            content="Payment succeeded with paid test data.",
                        ),
                    ]
                return {"messages": messages}

        return FakeAgent()

    target_namespace = {
        "INSTRUMENT_ID": "offline-instrument",
        "NETWORK": "ETHEREUM",
        "NETWORK_PREFERENCES": ["eip155:84532", "base-sepolia"],
        "PAYMENT_MANAGER_ARN": "offline-manager",
        "REGION": "us-west-2",
        "USER_ID": "offline-user",
        "build_agent": fake_build_agent,
    }
    with (
        patch(
            "bedrock_agentcore.payments.PaymentManager",
            return_value=fake_manager,
        ),
        patch(
            "bedrock_agentcore.payments.integrations.langgraph."
            "AgentCorePaymentsConfig",
            FakePaymentConfig,
        ),
        patch(
            "bedrock_agentcore.payments.integrations.langgraph."
            "AgentCorePaymentsMiddleware",
            FakePaymentMiddleware,
        ),
        redirect_stdout(StringIO()),
    ):
        exec(payment_integration_source, target_namespace)
        exec(target_source, target_namespace)
    assert isinstance(target_namespace["trace_payments"], FakePaymentMiddleware)
    assert target_namespace["trace_payment_config"].auto_session is True
    assert target_namespace["trace_payment_config"].auto_session_budget == "1.00"
    target = target_namespace["make_payment_target"](
        "offline prompt",
        "baseline",
    )
    cases = [
        ("automatic", "1.00", "paid"),
        ("explicit", "0.50", "paid"),
        ("explicit", "0.0001", "rejected"),
    ]
    safe_output_keys = {
        "scenario",
        "response",
        "http_evidence",
        "payment_outcome",
        "budget_usd",
        "remaining_usd",
        "spent_usd",
    }
    for index, (mode, budget, expected) in enumerate(cases):
        output = target(
            {
                "scenario": f"offline-{index}",
                "session_mode": mode,
                "budget_usd": budget,
                "request": "Use the paid test endpoint.",
            }
        )
        assert output["payment_outcome"] == expected
        assert set(output) == safe_output_keys
        if expected == "paid":
            assert output["http_evidence"] == {
                "status_code": 200,
                "body": '{"result": "paid test data"}',
                "truncated": False,
            }
            assert "payment-response" not in str(output["http_evidence"])
        else:
            assert output["http_evidence"] is None
    assert all(
        config["metadata"]["prompt_variant"] == "baseline"
        for config in invoke_configs
    )

    rule_evaluator_source = code_cell(
        notebook_source,
        "def payment_outcome_correct",
    )
    judge_evaluator_source = code_cell(notebook_source, "class ResponseGrade")
    assert "Payment outcome and AgentCore" in judge_evaluator_source
    assert "must not quote payment headers" in judge_evaluator_source
    evaluator_namespace = {
        "ChatBedrockConverse": StubJudgeModel,
        "Decimal": Decimal,
        "InvalidOperation": InvalidOperation,
        "MODEL_ID": "offline-model",
        "REGION": "us-west-2",
    }
    with redirect_stdout(StringIO()):
        exec(rule_evaluator_source, evaluator_namespace)
        exec(judge_evaluator_source, evaluator_namespace)
    outcome_check = evaluator_namespace["payment_outcome_correct"]
    budget_check = evaluator_namespace["budget_respected"]
    quality_check = evaluator_namespace["response_groundedness"]
    assert outcome_check(
        {"payment_outcome": "paid"},
        {"expected_payment_outcome": "paid"},
    )["score"]
    assert not outcome_check(
        {"payment_outcome": "rejected"},
        {"expected_payment_outcome": "paid"},
    )["score"]
    assert budget_check(
        {"budget_usd": "0.50"},
        {"spent_usd": "0.01"},
    )["score"]
    assert not budget_check(
        {"budget_usd": "0.50"},
        {"spent_usd": "0.51"},
    )["score"]
    assert quality_check(
        {"budget_usd": "0.50"},
        {
            "payment_outcome": "paid",
            "response": "Payment succeeded with paid test data.",
            "spent_usd": "0.01",
            "http_evidence": {
                "status_code": 200,
                "body": '{"result": "paid test data"}',
                "truncated": False,
            },
        },
        {"expected_payment_outcome": "paid"},
    )["score"]
    assert not quality_check(
        {"budget_usd": "0.50"},
        {
            "payment_outcome": "paid",
            "response": "Payment succeeded.",
            "spent_usd": "0.01",
            "http_evidence": None,
        },
        {"expected_payment_outcome": "paid"},
    )["score"]
    assert not quality_check({}, None, {})["score"]

    trace_source = code_cell(
        notebook_source,
        "trace_result = trace_agent.invoke",
    )

    @contextmanager
    def fake_tracing_context(**kwargs):
        assert kwargs["project_name"] == "offline-tracing-project"
        assert kwargs["enabled"] is True
        assert kwargs["client"] is fake_client
        yield

    class FakeTraceAgent:
        def __init__(self, payment_config) -> None:
            self.payment_config = payment_config

        def invoke(self, payload, config):
            assert payload["messages"][0][1] == "Use the paid test endpoint."
            assert config["recursion_limit"] == 8
            self.payment_config.payment_session_id = "offline-auto-session"
            return {
                "messages": [
                    SimpleNamespace(content="Paid HTTP response."),
                    SimpleNamespace(content="Payment succeeded."),
                ]
            }

    def fake_session_totals(payment_session_id, requested_budget):
        assert payment_session_id == "offline-auto-session"
        assert requested_budget == "1.00"
        return (
            Decimal("1.00"),
            Decimal("0.99"),
            Decimal("0.01"),
        )

    trace_payment_config = SimpleNamespace(payment_session_id=None)
    trace_namespace = {
        "Decimal": Decimal,
        "LANGSMITH_PROJECT": "offline-tracing-project",
        "NETWORK": "ETHEREUM",
        "request": "Use the paid test endpoint.",
        "session_totals": fake_session_totals,
        "trace_agent": FakeTraceAgent(trace_payment_config),
        "trace_payment_config": trace_payment_config,
    }
    with (
        patch("langsmith.Client", return_value=fake_client),
        patch("langsmith.tracing_context", fake_tracing_context),
        patch(
            "langchain_core.tracers.langchain.wait_for_all_tracers",
            return_value=None,
        ),
        redirect_stdout(StringIO()),
    ):
        exec(trace_source, trace_namespace)
    assert "offline-tracing-project" in fake_client.projects
    assert trace_namespace["trace_url"].startswith(
        "https://aws.smith.langchain.com/"
    )

    baseline_source = code_cell(notebook_source, "baseline_results =")
    audit_source = code_cell(notebook_source, "audit_results =")
    assert "max_concurrency=1" in baseline_source
    assert "max_concurrency=1" in audit_source


def main() -> None:
    expected = build_notebook()
    actual = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert notebook_source_signature(actual) == notebook_source_signature(
        expected
    ), "Agent notebook source differs. Restore or regenerate its cells."

    setup_expected = build_setup_notebook()
    setup_actual = SETUP_NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert notebook_source_signature(
        setup_actual
    ) == notebook_source_signature(setup_expected), (
        "Setup notebook source differs. Restore or regenerate its cells."
    )

    for notebook_source in (actual, setup_actual):
        notebook = nbformat.reads(notebook_source, as_version=4)
        nbformat.validate(notebook)
        assert all(cell.get("id") for cell in notebook.cells)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, f"notebook-cell-{cell.id}", "exec")

    for generated_source in (expected, setup_expected):
        generated_notebook = nbformat.reads(generated_source, as_version=4)
        assert all(
            not cell.get("outputs")
            for cell in generated_notebook.cells
            if cell.cell_type == "code"
        )

    verify_langsmith_workflow(actual)
    verify_cleanup_workflow(setup_actual)

    assert len(PAYMENT_ROLE_DEFINITIONS) == 4
    assert PAYMENT_ROLE_DEFINITIONS[MANAGEMENT_ROLE]["deny"] == [
        "bedrock-agentcore:ProcessPayment"
    ]
    assert "bedrock-agentcore:DeletePaymentSession" in (
        PAYMENT_ROLE_DEFINITIONS[MANAGEMENT_ROLE]["allow"]
    )

    not_found_error = botocore.exceptions.ClientError(
        {
            "Error": {
                "Code": "ResourceNotFoundException",
                "Message": "offline test",
            }
        },
        "GetPaymentSession",
    )
    assert is_not_found(not_found_error)

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
        (data_model, "ListPaymentSessions"): {
            "paymentManagerArn",
            "userId",
            "maxResults",
            "nextToken",
        },
        (data_model, "DeletePaymentSession"): {
            "paymentManagerArn",
            "paymentSessionId",
            "userId",
        },
        (data_model, "DeletePaymentInstrument"): {
            "paymentManagerArn",
            "paymentConnectorId",
            "paymentInstrumentId",
            "userId",
        },
        (control_model, "DeletePaymentConnector"): {
            "paymentManagerId",
            "paymentConnectorId",
            "clientToken",
        },
        (control_model, "DeletePaymentManager"): {
            "paymentManagerId",
            "clientToken",
        },
        (control_model, "DeletePaymentCredentialProvider"): {"name"},
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
        with redirect_stdout(StringIO()):
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
        "the middleware-first create_agent learning flow, payment tool "
        "registration, the standalone LangSmith trace walkthrough, and the "
        "grounded production-validation workflow, plus guarded cleanup in "
        "dependency order."
    )


if __name__ == "__main__":
    main()
