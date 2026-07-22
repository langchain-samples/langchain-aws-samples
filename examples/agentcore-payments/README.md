# AgentCore Payments with LangChain

Build a LangChain agent that can call an API that charges a small test payment.
AgentCore Payments handles the payment and enforces a spending limit. LangSmith
can trace the agent, model, and tool calls.

All example code and setup instructions are in this folder:

1. `setup_agentcore_payments.ipynb` creates the AWS resources and Coinbase
   test wallet once.
2. `agentcore_payments.ipynb` runs the payment-enabled agent.

You do not need to clone or run another repository.

## Will this spend real money?

The API payments use Base Sepolia, a test network, and free test USDC from a
faucet. Test USDC has no real-world value.

You may still see small AWS charges for Bedrock model calls, AgentCore API
calls, and storing the Coinbase connection in AWS. To avoid spending real
cryptocurrency:

- Keep `NETWORK=ETHEREUM`; this setup uses Base Sepolia, not Ethereum mainnet.
- Fund the wallet only with the Circle testnet faucet.
- Do not use Coinbase card, bank, or buy options.

## What the agent does

1. You ask the agent to get data from a paid URL.
2. The API replies that payment is required. This response is called HTTP 402
   or x402.
3. AgentCore Payments checks the spending limit, signs the test payment, and
   retries the request.
4. The API returns the data and the agent explains it.

The notebook shows a USD 1.00 limit, a USD 0.50 limit, and a limit that is
intentionally too small. The amounts are limits; the test calls do not spend
real USDC.

## Before you start

You need:

1. Python 3.11 or newer. Python 3.12 is the tested version.
2. [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for the
   recommended install path. A standard Python virtual environment also works.
3. The [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
   and local AWS credentials.
4. Access to the configured Anthropic Claude model in Amazon Bedrock.
5. Access to the AgentCore Payments preview in your AWS account.
6. AWS permission to create and assume IAM roles, pass a service role, and
   create AgentCore Payments and Secrets Manager resources.
7. A Coinbase account and a
   [Coinbase Developer Platform](https://portal.cdp.coinbase.com/) project.
8. Optional: an API key for the
   [AWS-region LangSmith instance](https://aws.smith.langchain.com).

Confirm that your AWS login works:

~~~bash
aws sts get-caller-identity
~~~

## 1. Install

Run these commands from this folder:

~~~bash
cp .env.example .env
uv sync --python 3.12
~~~

The committed `uv.lock` installs the tested package versions.

If you cannot use `uv`, use the fallback requirements:

~~~bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

The commands below use `uv run`. If you used the fallback, keep the virtual
environment active and remove `uv run` from each command.

## 2. Get the three Coinbase values

In the [Coinbase Developer Platform](https://portal.cdp.coinbase.com/):

1. Create or select a project.
2. Open **API Keys**, create a key, and save its API Key ID and API Key Secret.
3. Open **Wallets → ServerWallet** and save the Wallet Secret. Coinbase may
   show it only once, so store it in a password manager.
4. Open **Wallets → Embedded Wallet → Policies** and enable
   **Delegated Signing**. This lets AgentCore sign test payments for the wallet.

Never put these secret values in a notebook cell or commit them to Git.

## 3. Fill in `.env`

Open your local `.env`, review the provided defaults, and fill in the blank
setup values:

| Name | What to enter |
|---|---|
| `AWS_REGION` | Keep `us-west-2` unless your AgentCore Payments preview is in another region. |
| `MODEL_ID` | Keep the provided Bedrock model unless your account uses a different one. |
| `USER_ID` | Any label for this test user. The provided `test-user-001` is fine. |
| `LINKED_EMAIL` | The email you will use to sign in to the wallet page. |
| `COINBASE_API_KEY_ID` | The Coinbase API Key ID from step 2. |
| `COINBASE_API_KEY_SECRET` | The Coinbase API Key Secret from step 2. |
| `COINBASE_WALLET_SECRET` | The Coinbase ServerWallet secret from step 2. |
| `NETWORK` | Keep `ETHEREUM`; the notebook uses Base Sepolia testnet. |

Leave the following values blank. The setup notebook creates and fills them:

- `PAYMENT_MANAGER_ARN`
- `PAYMENT_MANAGER_ID`
- `PAYMENT_CONNECTOR_ID`
- `CREDENTIAL_PROVIDER_NAME`
- `CREDENTIAL_PROVIDER_ARN`
- `INSTRUMENT_ID`
- `WALLET_ADDRESS`
- The four values ending in `ROLE_ARN`

`.env` is ignored by Git. Do not commit it.

If your team already gave you an existing funded test wallet and its three
required values, you can skip the setup notebook. Fill in
`PAYMENT_MANAGER_ARN`, `USER_ID`, and `INSTRUMENT_ID`, make sure `NETWORK`
matches that wallet, and continue to step 5.

## 4. Create and fund the test wallet

Open the setup notebook:

~~~bash
uv run jupyter lab setup_agentcore_payments.ipynb
~~~

Run its cells from top to bottom. It creates four limited-purpose AWS roles,
the AgentCore payment configuration, and an embedded Coinbase test wallet. It
writes the generated IDs to `.env` without printing your Coinbase secrets.

The notebook then prints a wallet address and a WalletHub URL:

1. Open the [Circle Faucet](https://faucet.circle.com/), choose
   **Base Sepolia**, and send test USDC to the printed wallet address.
2. Open the printed WalletHub URL, sign in with `LINKED_EMAIL`, and approve
   signing if asked.
3. Return to the notebook and rerun the balance cell. Continue when it shows a
   balance greater than zero.

These two browser actions cannot be automated by the notebook.

## 5. Optional: enable LangSmith tracing

To record the agent, model, and tool calls, add these values to `.env`:

~~~bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com
LANGSMITH_API_KEY=<your-aws-region-langsmith-api-key>
LANGSMITH_PROJECT=agentcore-payments
~~~

Use a key from the AWS-region LangSmith site. A key from the standard
`smith.langchain.com` site will not work with this endpoint.

Tracing is optional. If no key is set, the notebook disables tracing and the
agent still runs. Traces may contain prompts, URLs, API responses, and model
output, so use test data only.

## 6. Run the agent

~~~bash
uv run jupyter lab agentcore_payments.ipynb
~~~

Run the cells from top to bottom. The notebook checks the required settings
before it creates a payment session or calls the model. Payment sessions expire
after 60 minutes.

With LangSmith enabled, open the `agentcore-payments` project to compare the
normal payment, fixed-limit, and insufficient-limit runs.

## Offline check

This check does not read `.env`, call AWS, invoke a model, or access a paid API:

~~~bash
uv run python verify.py
~~~

It confirms that both notebooks match their source files, the setup roles are
defined, and the LangChain agent and payment integration can be created.

## Cleanup

Payment sessions expire automatically, but the setup resources remain. The
last section of `setup_agentcore_payments.ipynb` lists the cleanup commands and
the required order. Delete the wallet instrument first, then the connection,
payment configuration, stored Coinbase connection, and four example roles.

## Troubleshooting

- **A required value is missing:** run the setup notebook and finish the
  non-zero balance check.
- **Setup says access denied:** confirm your AWS permissions, preview access,
  and region.
- **The payment is rejected:** confirm the wallet has Base Sepolia test USDC
  and that signing was approved in both the Coinbase project and WalletHub.
- **No LangSmith trace appears:** confirm the API key and endpoint are from the
  same AWS-region LangSmith account.
- **The paid test URL fails:** public test services can change. Use another
  Base Sepolia x402 test URL and keep the limit small.

## Before using this in production

- Keep each user's wallet and spending session separate.
- Use small limits and short expiration times.
- Never let the model choose wallet, user, or payment IDs.
- Treat data returned by paid APIs as untrusted.
- Require human approval before using a real network or larger limits.

## Source and license

This example is adapted from the
[AWS AgentCore Payments middleware notebook](https://github.com/awslabs/agentcore-samples/blob/496c79e72b2a2e7318c8230e6707460bbed0883a/06-workshops/13-AgentCore-payments/00-getting-started/01-agents-payments-and-limits/langgraph_payment_agent_middleware.ipynb)
and the
[AWS AgentCore Payments setup notebook](https://github.com/awslabs/agentcore-samples/blob/3a8d5352daeeaf17d17e0b724e8927fd917f5f79/06-workshops/13-AgentCore-payments/00-getting-started/00-setup-agentcore-payments/setup_agentcore_payments.ipynb).
See [NOTICE](./NOTICE) and [LICENSE-APACHE](./LICENSE-APACHE) for the upstream
license and modification notice.
