# AgentCore Payments with LangChain

Build a LangChain agent that can call an x402 paid API while Amazon Bedrock
AgentCore Payments handles payment authorization and enforces a spending
limit. Trace the agent, model, and tool flow in the AWS-region LangSmith
instance.

This is a focused, single-agent example. It uses LangChain's
`create_agent()` API, which runs on LangGraph, and attaches
`AgentCorePaymentsMiddleware` once for automatic HTTP 402 handling.

## What the agent does

~~~text
User asks for data from a paid URL
  -> Bedrock model calls the http_request tool
  -> API responds with HTTP 402 Payment Required
  -> AgentCore Payments creates or checks a budgeted session
  -> middleware signs the payment request and retries
  -> API returns the paid data
  -> model explains the result
~~~

The notebook demonstrates:

1. An automatically created session capped at USD 1.00.
2. An explicit session capped at USD 0.50.
3. A USD 0.0001 session that is intentionally too small, proving that the
   service—not the model—enforces the spending limit.

## Cost and testnet warning

The payment examples default to **Base Sepolia testnet** and use faucet USDC
with no real-world value. They do not spend real USDC.

Running the example can still create small, ordinary AWS charges:

- Amazon Bedrock model inference.
- AgentCore, CloudWatch, or related resources created by the prerequisite
  setup tutorial.

AgentCore Payments is a preview service. APIs, availability, and pricing may
change. Do not change the example to a mainnet network unless you deliberately
intend to use assets with real-world value.

## Prerequisites

Before running the notebook, you need:

1. Python 3.11 or newer. Python 3.12 is the tested version.
2. AWS credentials configured locally. Verify them with
   `aws sts get-caller-identity`.
3. Access to the configured Anthropic Claude model in Amazon Bedrock.
4. An AWS account with access to the AgentCore Payments preview.
5. An AgentCore Payment Manager, payment instrument, and testnet-funded
   wallet. Complete the upstream
   [AgentCore Payments setup tutorial](https://github.com/awslabs/agentcore-samples/tree/main/06-workshops/13-AgentCore-payments/00-getting-started/00-setup-agentcore-payments)
   first.
6. Recommended: an account and API key for the
   [AWS-region LangSmith instance](https://aws.smith.langchain.com).

The upstream setup creates the identifiers this example expects:
`PAYMENT_MANAGER_ARN`, `USER_ID`, and `INSTRUMENT_ID`. It also explains
how to fund the instrument with free testnet USDC and enable delegated
signing.

## Install

From this directory:

~~~bash
cp .env.example .env
uv sync --python 3.12
~~~

`uv.lock` is committed so everyone installs the same tested dependency set.
If `uv` is unavailable, use a standard virtual environment and
`requirements.txt` as a fallback:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

## Configure

Fill in these values in your local `.env`:

| Variable | Purpose |
|---|---|
| `AWS_REGION` | Region containing the AgentCore Payments resources and Bedrock model. |
| `MODEL_ID` | Bedrock model ID used by the agent. |
| `PAYMENT_MANAGER_ARN` | Payment Manager created by the setup tutorial. |
| `USER_ID` | End-user ID associated with the payment instrument. |
| `INSTRUMENT_ID` | Testnet-funded payment instrument. |
| `NETWORK` | `ETHEREUM` for Base Sepolia, the default, or `SOLANA` for Solana Devnet. |
| `PAID_API_URL` | Optional testnet x402 endpoint to call. |

To enable LangSmith tracing, also set:

~~~bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com
LANGSMITH_API_KEY=<your-aws-region-langsmith-api-key>
LANGSMITH_PROJECT=agentcore-payments
~~~

Use the API endpoint above, not the LangSmith UI URL. A key from the default
`smith.langchain.com` instance will not authenticate against the AWS-region
instance.

Never commit `.env`. Do not add Payment Manager ARNs, instrument IDs,
session IDs, wallet addresses, or user IDs to LangSmith tags or metadata.
Traces can contain prompts, requested URLs, API responses, and model output.
Use test data here and configure appropriate redaction and retention controls
before tracing sensitive production workloads.

## Run

~~~bash
uv run jupyter lab agentcore_payments.ipynb
~~~

Run the cells from top to bottom. The notebook validates configuration before
creating sessions or invoking the model.

Payment sessions expire automatically after 60 minutes. The notebook prints
only the configured budget and remaining amount; it does not print payment
credentials or identifiers.

For an offline check that does not read `.env`, call AWS, invoke a model, or
access a paid endpoint, run:

~~~bash
uv run python verify.py
~~~

## What LangSmith captures

With tracing enabled, the project shows the agent run, Bedrock model calls,
tool calls, timing, and final responses. Runs are named and tagged by scenario
so the automatic, explicit-budget, and insufficient-budget traces are easy to
compare.

In `bedrock-agentcore` 1.18.1, session creation and payment-header signing
occur inside the payment middleware and are not emitted as separate LangSmith
child spans. Use the AgentCore and AWS observability surfaces when you need
service-level payment diagnostics.

## Production notes

- Create a new middleware instance per concurrent user or payment session.
  Automatic session creation writes the session ID onto that middleware's
  configuration object.
- Set conservative budgets and short expirations.
- Keep the payment instrument and session associated with the authenticated
  application user; never let the model select raw identifiers.
- Treat paid endpoint responses as untrusted input.
- Add an explicit approval policy before enabling mainnet or materially larger
  budgets.

## Troubleshooting

- **Missing configuration:** confirm the three payment identifiers were copied
  from the upstream setup tutorial into your local `.env`.
- **Access denied:** confirm AWS credentials, preview access, IAM permissions,
  region, and Bedrock model access.
- **Payment rejected:** confirm the wallet has testnet USDC and delegated
  signing is enabled for the selected wallet provider.
- **No LangSmith trace:** confirm the AWS-region endpoint and API key are from
  the same LangSmith tenant.
- **Paid endpoint unavailable:** public test endpoints can change; substitute
  another testnet x402 endpoint and keep the budget small.

## Source and license

This example is adapted from the
[AWS AgentCore Payments middleware notebook](https://github.com/awslabs/agentcore-samples/blob/496c79e72b2a2e7318c8230e6707460bbed0883a/06-workshops/13-AgentCore-payments/00-getting-started/01-agents-payments-and-limits/langgraph_payment_agent_middleware.ipynb).
See [NOTICE](./NOTICE) and [LICENSE-APACHE](./LICENSE-APACHE) for the upstream
license and modification notice.
