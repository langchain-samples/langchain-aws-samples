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

## Journey at a glance

1. Confirm your AWS account has AgentCore Payments preview access and Bedrock
   model access.
2. Install the example.
3. Create or sign in to Coinbase Developer Platform and collect three values.
4. Add those values to `.env`.
5. Run the setup notebook, fund the wallet with free test USDC, and approve
   test-payment signing.
6. Optionally create a LangSmith account and enable tracing.
7. Run the agent notebook and check all three spending-limit examples.
8. Delete the test resources when finished.

No other repository is involved.

## Accounts at a glance

| Account or site | Needed? | What to do |
|---|---|---|
| AWS | Required | Use an account with AgentCore Payments preview access. Step 1 explains this. |
| Coinbase Developer Platform | Required for local wallet setup | Create an account and project in step 3. No payment method or crypto purchase is needed. |
| AWS-region LangSmith | Optional | Create or join an account in step 6 only if you want traces. |
| Circle Faucet | No account | Paste the test wallet address into the public faucet. |
| WalletHub | No separate account | Sign in with `LINKED_EMAIL` when the setup notebook gives you the link. |
| Anthropic | No account | Bedrock provides the Claude model, so no Anthropic API key is needed. |

## 1. Confirm AWS access

If your company or the AgentCore Payments team provided an AWS account, use
that account. If you have no AWS account, follow the
[AWS account creation guide](https://aws.amazon.com/resources/create-account/).
AWS signup may require a payment method, and AWS usage can incur charges.

A newly created AWS account does **not** automatically have AgentCore Payments
preview access. Ask the AgentCore Payments team or your AWS contact to:

1. Enable the preview for your AWS account.
2. Confirm the AWS region to use. This example defaults to `us-west-2`.
3. Give your non-root AWS user or role permission to create and assume IAM
   roles, pass a service role, and create AgentCore Payments and Secrets
   Manager resources.

Install the
[AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
and configure it using your team's normal AWS sign-in method. If your team uses
AWS IAM Identity Center, follow the
[AWS CLI sign-in guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html).
Do not create or use root-account access keys.

Confirm that the terminal you will use for Jupyter can access AWS:

~~~bash
aws sts get-caller-identity
~~~

Finally, confirm that the Bedrock model named by `MODEL_ID` is available in the
same region. See
[Access Amazon Bedrock foundation models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html).

## 2. Install

You need Python 3.11 or newer; Python 3.12 is the tested version. Install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run these
commands from the repository root:

~~~bash
cd examples/agentcore-payments
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

## 3. Create the Coinbase account and get three values

1. Open the [Coinbase Developer Platform](https://portal.cdp.coinbase.com/).
2. If you are new to Coinbase, choose the sign-up option and complete the
   email verification. Otherwise, sign in.
3. Create or select a Developer Platform project.
4. Open **API Keys**, create a key, and save its API Key ID and API Key Secret.
5. Open **Wallets → ServerWallet** and save the Wallet Secret. Coinbase may
   show it only once, so store it in a password manager.
6. Open **Wallets → Embedded Wallet → Policies** and enable
   **Delegated Signing**. This lets AgentCore sign test payments for the wallet.

You do not need to add a payment method or buy cryptocurrency for this example.

Never put these secret values in a notebook cell or commit them to Git.

## 4. Fill in `.env`

Open your local `.env`, review the provided defaults, and fill in the blank
setup values:

| Name | What to enter |
|---|---|
| `AWS_REGION` | Keep `us-west-2` unless your AgentCore Payments preview is in another region. |
| `MODEL_ID` | Keep the provided Bedrock model unless your account uses a different one. |
| `USER_ID` | Any label for this test user. The provided `test-user-001` is fine. |
| `LINKED_EMAIL` | The email you will use to sign in to the wallet page. |
| `COINBASE_API_KEY_ID` | The Coinbase API Key ID from step 3. |
| `COINBASE_API_KEY_SECRET` | The Coinbase API Key Secret from step 3. |
| `COINBASE_WALLET_SECRET` | The Coinbase ServerWallet secret from step 3. |
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
matches that wallet, and continue to step 6.

## 5. Create and fund the test wallet

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

The Circle faucet is public and does not require an account. WalletHub uses
`LINKED_EMAIL` for sign-in and does not require a separate account. These two
browser actions cannot be automated by the notebook.

## 6. Optional: create a LangSmith account and enable tracing

Skip this section if you do not want traces. The agent works without LangSmith.

To enable tracing:

1. Open the [AWS-region LangSmith site](https://aws.smith.langchain.com/).
2. Create an account, sign in, or join your team's existing workspace.
3. Open settings, create an API key, and save it in your password manager.
4. Add the following values to `.env`:

~~~bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com
LANGSMITH_API_KEY=<your-aws-region-langsmith-api-key>
LANGSMITH_PROJECT=agentcore-payments
~~~

Use a key from the AWS-region LangSmith site. A key from the standard
`smith.langchain.com` site will not work with this endpoint.

The `agentcore-payments` project is created when its first trace arrives. If no
key is set, the notebook disables tracing and the agent still runs. Traces may
contain prompts, URLs, API responses, and model output, so use test data only.

## 7. Run the agent

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

## 8. Cleanup

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
