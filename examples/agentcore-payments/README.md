# AgentCore Payments with LangChain

Build a LangChain agent that can call an x402 paid API while Amazon Bedrock
AgentCore Payments handles payment authorization and enforces a spending
limit. Trace the agent, model, and tool flow in the AWS-region LangSmith
instance.

This is a focused, single-agent example. It uses LangChain's
`create_agent()` API, which runs on LangGraph, and attaches
`AgentCorePaymentsMiddleware` once for automatic HTTP 402 handling.

Everything needed is in this folder:

1. `setup_agentcore_payments.ipynb` creates the AgentCore resources and
   embedded Coinbase testnet wallet once.
2. `agentcore_payments.ipynb` runs and traces the payment-enabled agent.

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
- AgentCore Payments API calls and managed credential storage created by the
  setup notebook.

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
5. Permission to create and assume IAM roles, pass a service role, and create
   AgentCore Payments and Secrets Manager resources.
6. For the local setup path, a Coinbase account and a
   [Coinbase Developer Platform](https://portal.cdp.coinbase.com/) project.
7. Recommended: an account and API key for the
   [AWS-region LangSmith instance](https://aws.smith.langchain.com).

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

## One-time local setup

If your team already supplied a Payment Manager ARN, user ID, and funded
instrument ID, put those values in `.env` and skip directly to
**Configure LangSmith**. Otherwise, follow the local setup below.

### 1. Get Coinbase CDP credentials

In the [CDP Portal](https://portal.cdp.coinbase.com/):

1. Create or select a project.
2. Open **API Keys**, create a key, and retain its API Key ID and API Key
   Secret.
3. Open **Wallets → ServerWallet** and retain the Wallet Secret. Coinbase may
   show it only once, so save it in a password manager.
4. Open **Wallets → Embedded Wallet → Policies** and enable
   **Delegated Signing**.

Add only these setup inputs to your local `.env`:

| Variable | Purpose |
|---|---|
| `AWS_REGION` | Region where AgentCore Payments and Bedrock are available. |
| `USER_ID` | Application-level label for the wallet owner; the default is `test-user-001`. |
| `LINKED_EMAIL` | Email used to sign in to the embedded-wallet experience. |
| `COINBASE_API_KEY_ID` | CDP API Key ID. |
| `COINBASE_API_KEY_SECRET` | CDP API Key Secret. |
| `COINBASE_WALLET_SECRET` | CDP ServerWallet secret. |
| `NETWORK` | Keep this as `ETHEREUM`; the local setup creates a Base Sepolia wallet. |

Do not fill `PAYMENT_MANAGER_ARN`, `PAYMENT_MANAGER_ID`, or
`INSTRUMENT_ID`. The setup notebook generates them.

### 2. Create the AWS resources and wallet

~~~bash
uv run jupyter lab setup_agentcore_payments.ipynb
~~~

Run the setup cells in order. They create four scoped IAM roles, a Coinbase
credential provider, Payment Manager, connector, and embedded wallet. Resource
identifiers are written back into the same local `.env` without printing
credential values.

### 3. Complete the two browser actions

When the setup notebook prints the wallet address and WalletHub URL:

1. Open [Circle Faucet](https://faucet.circle.com/), select **Base Sepolia**,
   and fund the address with testnet USDC.
2. Open WalletHub, sign in with `LINKED_EMAIL`, and grant signing permission
   if prompted.
3. Rerun the balance cell. A non-zero testnet balance completes setup.

Use the faucet only. Do not use card, bank, or on-ramp options for this
tutorial because those can involve assets with real-world value.

The setup notebook fills the three values consumed by the agent:
`PAYMENT_MANAGER_ARN`, `USER_ID`, and `INSTRUMENT_ID`.

## Configure LangSmith

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

## Run the payment agent

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

The setup notebook uses separate roles for control-plane and management
operations. For notebook convenience, the payment agent itself uses your
active AWS credential chain; that identity must be allowed to create sessions
and process payments. Production applications should preserve role separation
between the backend that creates budgets and the runtime that spends them.

## Cleanup

Payment sessions expire automatically, but setup resources persist. After
testing, use the cleanup order documented at the end of
`setup_agentcore_payments.ipynb`: instrument, connector, manager, credential
provider/managed secret, then the four example-specific IAM roles.

## Troubleshooting

- **Missing configuration:** run `setup_agentcore_payments.ipynb` first and
  confirm it completed the non-zero balance check.
- **Setup access denied:** confirm AgentCore Payments preview access and the
  IAM permissions listed under Prerequisites.
- **Access denied:** confirm AWS credentials, preview access, IAM permissions,
  region, and Bedrock model access.
- **Payment rejected:** confirm the wallet has testnet USDC and delegated
  signing is enabled in the CDP project and WalletHub.
- **No LangSmith trace:** confirm the AWS-region endpoint and API key are from
  the same LangSmith tenant.
- **Paid endpoint unavailable:** public test endpoints can change; substitute
  another testnet x402 endpoint and keep the budget small.

## Source and license

This example is adapted from the
[AWS AgentCore Payments middleware notebook](https://github.com/awslabs/agentcore-samples/blob/496c79e72b2a2e7318c8230e6707460bbed0883a/06-workshops/13-AgentCore-payments/00-getting-started/01-agents-payments-and-limits/langgraph_payment_agent_middleware.ipynb).
The local setup is adapted from the
[AWS AgentCore Payments setup notebook](https://github.com/awslabs/agentcore-samples/blob/3a8d5352daeeaf17d17e0b724e8927fd917f5f79/06-workshops/13-AgentCore-payments/00-getting-started/00-setup-agentcore-payments/setup_agentcore_payments.ipynb).
See [NOTICE](./NOTICE) and [LICENSE-APACHE](./LICENSE-APACHE) for the upstream
license and modification notice.
