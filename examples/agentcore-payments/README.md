# AgentCore Payments with LangChain

Build a LangChain agent that can call an API that charges a small test payment.
AgentCore Payments handles the payment and enforces a spending limit. LangSmith
traces the workflow, stores three policy cases as a dataset, scores the results,
and compares two prompt variants.

All example code and Coinbase quick-start instructions are in this folder:

1. `setup_agentcore_payments.ipynb` creates the AWS resources and Coinbase
   test wallet once.
2. `agentcore_payments.ipynb` runs the payment-enabled agent.

For the Coinbase path, you do not need to clone or run another repository.
Follow this README in order. Do not start `agentcore_payments.ipynb` until you
have completed either the included setup notebook through its balance check or
an existing AgentCore Payments setup, and finished the LangSmith account setup
in step 6 below.

## Will this spend real money?

The API payments use Base Sepolia, a test network, and free test USDC from a
faucet. Test USDC has no real-world value.

You may still see small charges for Bedrock model calls, the managed AWS secret,
and LangSmith usage beyond your plan's included allowance. The agent notebook
runs one tracing walkthrough, six experiment cases, and six judge evaluations.
To avoid spending real cryptocurrency:

- Keep `NETWORK=ETHEREUM`; this setup uses Base Sepolia, not Ethereum mainnet.
- Fund the wallet only with the Circle testnet faucet.
- Do not use Coinbase card, bank, or buy options.

The tracing walkthrough and four successful experiment cases spend only
faucet test USDC. The exact test amount depends on the paid endpoint's current
price.

## What the agent does

1. You ask the agent to get data from a paid URL.
2. The API replies that payment is required. This response is called HTTP 402
   or x402.
3. AgentCore Payments checks the spending limit, signs the test payment, and
   retries the request.
4. The API returns the data and the agent explains it.

The notebook evaluates a USD 1.00 limit, a USD 0.50 limit, and a limit that is
intentionally too small. These are infrastructure-enforced limits. Successful
calls spend faucet USDC, which has no real-world value.

## Journey at a glance

The numbered quick start below uses Coinbase CDP:

1. Confirm your AWS account has AgentCore Payments preview access and Bedrock
   model access.
2. Install the example.
3. Create or sign in to Coinbase Developer Platform and collect three values.
4. Add those values to `.env`.
5. Run the setup notebook, fund the wallet with free test USDC, and approve
   test-payment signing.
6. Create or join an AWS-region LangSmith account and enable tracing and evals.
7. Run the agent notebook to build the middleware and open-source agent,
   inspect one trace, then validate two prompt versions like production
   candidates.
8. Delete the test resources when finished.

This follows the Build and Test portions of LangChain's
[Agent Development Lifecycle](https://www.langchain.com/blog/the-agent-development-lifecycle).
It stops before deployment and shows how production traces and feedback would
feed the next evaluation cycle. AgentCore spending limits enforce payment
safety; LangSmith provides visibility and evidence.

## Accounts at a glance

| Account or site | Needed? | What to do |
|---|---|---|
| AWS | Required | Use an account with AgentCore Payments preview access. Step 1 explains this. |
| Coinbase Developer Platform | Required for the Coinbase quick start | Create an account and project in step 3. No payment method or crypto purchase is needed. |
| AWS-region LangSmith | Required | Create or join an account in step 6 and enable tracing for the workshop. |
| Circle Faucet | No account | Paste the test wallet address into the public faucet. |
| WalletHub | No separate account | Sign in with `LINKED_EMAIL` when the setup notebook gives you the link. |
| Anthropic | No account | Bedrock provides the Claude model, so no Anthropic API key is needed. |

## Choose a payment provider

This workshop uses Coinbase CDP as its tested, self-contained setup path. After
AgentCore Payments is configured, the agent only needs a payment manager, user,
instrument, and matching network. It can therefore use any wallet or payment
instrument supported and configured through AgentCore Payments.

Choose one path:

- **Coinbase CDP:** Follow the numbered quick start below and use the included
  setup notebook.
- **Stripe/Privy or multiple providers:** Complete steps 1 and 2 below, then
  use the AWS
  [AgentCore Payments setup samples](https://github.com/awslabs/agentcore-samples/tree/main/01-features/08-agents-that-transact/00-getting-started/00-setup-agentcore-payments).
  Provider-specific account guidance is in its
  [providers folder](https://github.com/awslabs/agentcore-samples/tree/main/01-features/08-agents-that-transact/00-getting-started/00-setup-agentcore-payments/providers).
- **Existing AgentCore Payments setup:** Skip this example's setup notebook.

For either alternative path, add its `PAYMENT_MANAGER_ARN`, `USER_ID`,
`INSTRUMENT_ID`, and matching `NETWORK` to `.env`, then continue at step 6.
The agent, LangSmith tracing, dataset, and evaluation workflow are unchanged.

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

## 3. Coinbase quick start: create the account and get three values

1. Open the [Coinbase Developer Platform](https://portal.cdp.coinbase.com/).
2. If you are new to Coinbase, choose the sign-up option and complete the
   email verification. Otherwise, sign in.
3. Use the project the portal opens for you. Coinbase may create a default
   project automatically, so you might not see a separate project-creation
   screen. If the portal shows a project switcher, create or select a project.
4. Open **API Keys → Secret API keys**. Do **not** choose **Client API Key**;
   client keys are for browser and mobile apps, while AgentCore needs a
   server-side secret key.
5. Create a Secret API Key with these settings:
   - Nickname: `agentcore-payments-tutorial`
   - IP allowlist: select **Opt-out of IP allowlisting**. AgentCore calls
     Coinbase from AWS, so a private laptop address such as `192.168...` will
     not work. If your organization requires an allowlist, ask the AgentCore
     Payments team for supported outbound IPs instead of guessing.
   - Advanced settings: keep the default **Ed25519** signing algorithm.
6. Save the API Key ID and API Key Secret when they are shown.
7. Open
   [**Wallets → Non-custodial Wallet → Security**](https://portal.cdp.coinbase.com/wallets/non-custodial/security).
   Do not choose Custodial Wallet or Agentic Wallet.
8. Under **Wallet Secret**, generate the secret and save it immediately in a
   password manager. Coinbase may show it only once.
9. On the same **Security** page, enable **Delegated Signing**. This lets
   AgentCore sign test payments for the wallet.

You do not need to add a payment method or buy cryptocurrency for this example.

See Coinbase's
[authentication guide](https://docs.cdp.coinbase.com/get-started/authentication/overview)
for the difference between Secret and Client API Keys.

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
| `COINBASE_API_KEY_ID` | The ID from the Coinbase Secret API Key created in step 3. |
| `COINBASE_API_KEY_SECRET` | The secret from that Coinbase Secret API Key. |
| `COINBASE_WALLET_SECRET` | The Non-custodial Wallet Secret from step 3. |
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

If you are using an AgentCore Payments setup created elsewhere, skip the setup
notebook. Fill in
`PAYMENT_MANAGER_ARN`, `USER_ID`, and `INSTRUMENT_ID`, make sure `NETWORK`
matches that wallet, and continue to step 6.

## 5. Create and fund the test wallet

Open the setup notebook:

~~~bash
uv run jupyter lab setup_agentcore_payments.ipynb
~~~

Run the cells from the top **through Step 9**, including the cell that prints
the wallet address and WalletHub URL. **Stop before Step 10; do not use Run
All yet.** By this point, the notebook has created four limited-purpose AWS
roles, the AgentCore payment configuration, and an embedded Coinbase test
wallet. It also writes the generated IDs to `.env` without printing your
Coinbase secrets.

Complete the required browser steps before continuing:

1. Open the [Circle Faucet](https://faucet.circle.com/), choose
   **Base Sepolia**, and send 20 test USDC to the printed wallet address.
2. Open the printed WalletHub URL, sign in with `LINKED_EMAIL`, and approve
   signing if asked.
3. Return to the notebook and run Step 10. Continue when it shows a balance
   greater than zero. If the balance is still zero, wait briefly and rerun
   Step 10.

The Circle faucet is public and does not require an account. WalletHub uses
`LINKED_EMAIL` for sign-in and does not require a separate account. These two
browser actions cannot be automated by the notebook.

## 6. Create a LangSmith account and enable tracing and evals

LangSmith is a required part of this workshop. You will use it to inspect
traces, create a dataset, run evaluators, and compare prompt experiments.

1. Open the [AWS-region LangSmith site](https://aws.smith.langchain.com/).
2. If you are new to LangSmith, create an account and workspace. Otherwise,
   sign in or accept your team's workspace invitation.
3. In that AWS-region workspace, open settings, create an API key, and save it
   in your password manager.
4. Add the following values to `.env`:

~~~bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com
LANGSMITH_API_KEY=<your-aws-region-langsmith-api-key>
LANGSMITH_PROJECT=agentcore-payments
~~~

Use a key from the AWS-region LangSmith site. A key from the standard
`smith.langchain.com` site will not work with this endpoint.

The agent notebook creates two separate LangSmith views:

- **Tracing Projects → the value of `LANGSMITH_PROJECT`** contains the
  standalone automatic-payment walkthrough from Step 4. The provided value is
  `agentcore-payments`.
- **Datasets & Experiments → `agentcore-payments-policy-evals`** contains the
  six evaluation runs and their traces.

Experiment runs stay with their dataset experiments; they are not copied into
the standalone tracing project. The notebook stops before running the agent if
these LangSmith settings are missing or tracing is disabled. Traces and
experiments may contain prompts, URLs, API responses, and model output, so use
test data only.

## 7. Run the agent

Before continuing, confirm both prerequisites are complete:

- Step 10 of `setup_agentcore_payments.ipynb` shows a non-zero testnet balance,
  or your alternative AgentCore Payments instrument is funded and ready.
- You created or joined the AWS-region LangSmith account and added all four
  LangSmith settings from step 6 to `.env`.

~~~bash
uv run jupyter lab agentcore_payments.ipynb
~~~

Run the cells in order, one section at a time. **Do not use Run All.** The
notebook checks the required settings before it creates a dataset, payment
session, or model call. It then walks you through this required LangSmith
workflow:

1. Complete Step 2 to build the AgentCore Payments configuration, helper for
   fresh limited sessions, and middleware. This is the core AWS integration.
2. Complete Step 3 to create the LangChain agent and see that it runs on a
   compiled LangGraph runtime.
3. Run Step 4, then stop. Open the direct LangSmith trace link printed by the
   notebook and follow the run from model call to HTTP 402, test payment,
   successful retry, and final response.
4. In Step 5, treat the agent as a production candidate and create three
   reusable acceptance cases in a LangSmith dataset.
5. Complete Steps 6–7 to build the repeatable target, two policy evaluators,
   and an evidence-grounded Bedrock judge.
6. Run Step 8 to validate the baseline. Stop and inspect its rows, evaluator
   feedback, and traces under **Datasets & Experiments**.
7. Run Step 9 to harden the reporting boundary. The model summarizes API data
   while the application keeps AgentCore session totals authoritative; all
   evaluator scores should remain passing.
8. Complete Step 10: compare the two experiment names printed by the current
   run and make a release decision. Older experiments remain available.

Each insufficient-budget experiment intentionally prints insufficient-budget
log lines. Those messages prove the limit blocked the payment; they are not
notebook errors.

Expected policy results:

| Case | Expected outcome | Required safety result |
|---|---|---|
| Automatic USD 1.00 | Payment succeeds | Spending stays at or below USD 1.00 |
| Explicit USD 0.50 | Payment succeeds | Spending stays at or below USD 0.50 |
| Explicit USD 0.0001 | Payment is rejected | The limit is not exceeded |

All three evaluator scores should be `True` for every row in both experiments.
The insufficient-budget case prints payment-error logs by design; those logs
show that AgentCore enforced the limit.

The notebook runs each experiment sequentially and prints its LangSmith link
and score summary. The second experiment changes only the prompt, letting you
compare grounded response quality, latency, tokens, and model cost without
changing the payment policy. Payment sessions expire after 60 minutes.

In this workshop:

- **Middleware** adds the paid HTTP tool and enforces the AgentCore payment
  flow around the agent.
- `create_agent()` is the LangChain API and returns the LangGraph runtime.
- A **trace** records one full agent trajectory so you can understand model,
  tool, payment, and final-response behavior.
- A **dataset** preserves three reusable inputs and expected payment outcomes
  so future changes are tested against the same behavior.
- A **target** is the application LangSmith runs for every dataset row.
- An **evaluator** turns a payment rule or quality criterion into a score.
- An **experiment** runs one version over the complete dataset so versions can
  be compared without changing the test cases.
- A **tracing project** groups normal application runs. In this workshop it
  contains the separate Step 4 walkthrough.

## Offline check

This check does not read `.env`, call AWS, invoke a model, or access a paid API:

~~~bash
uv run python verify.py
~~~

It confirms that both notebooks match their source files, the setup roles are
defined, the LangChain payment integration can be created, and the trace,
dataset, and grounded evaluator paths are wired correctly. You can run it
before or after the workshop: it ignores normal saved cell outputs but still
detects changed code or instructions.

## 8. Cleanup

Payment sessions expire automatically, but the other setup resources remain.
If you used the Coinbase quick start, return to the **Cleanup** section at the
bottom of `setup_agentcore_payments.ipynb` when you are completely finished:

1. In the first cleanup cell, set `CLEANUP_CONFIRMATION` to
   `DELETE AGENTCORE PAYMENTS TEST RESOURCES`, then run the cell. It deletes
   the workshop payment sessions, wallet instrument, connector, manager,
   stored Coinbase credential, and four example IAM roles in the required
   order. It then blanks their generated IDs in `.env`.
2. In Coinbase Developer Platform, open **API Keys → Secret API keys** and
   revoke the dedicated `agentcore-payments-tutorial` key. The notebook cannot
   safely do this for you.
3. After revoking the key, optionally set `LOCAL_CLEAR_CONFIRMATION` to
   `CLEAR LOCAL COINBASE VALUES` in the final cleanup cell and run it. This
   blanks the local Coinbase values and linked email in `.env`.

Each code cell is locked until its exact phrase is entered. The AWS cleanup is
safe to rerun: an already deleted resource is skipped, and generated IDs stay
in `.env` if cleanup stops before all AWS deletions finish. Fix the reported
permission or dependency issue, then rerun the same cell. No cleanup cell
prints resource IDs or secrets.

LangSmith data is intentionally kept so you can return to the traces and
compare later agent versions. If you eventually want to remove it, do that
from the LangSmith UI rather than the AWS setup notebook.

If you used an AWS provider setup sample instead, follow that sample's cleanup
instructions for the resources it created.

## Troubleshooting

- **A required value is missing:** run the setup notebook and finish the
  non-zero balance check.
- **Setup says access denied:** confirm your AWS permissions, preview access,
  and region.
- **The payment is rejected:** confirm the wallet has Base Sepolia test USDC
  and that signing was approved in both the Coinbase project and WalletHub.
- **The dataset exists but no tracing project appears:** run Step 4 of the
  agent notebook. Creating experiments does not create the separate project
  named by `LANGSMITH_PROJECT`.
- **Step 4 cannot find its trace:** confirm the API key and endpoint are from
  the same AWS-region LangSmith account and that the key can create projects.
  Rerun Step 4; it safely reuses the project and creates a new walkthrough.
- **The dataset cell fails:** confirm your LangSmith key can create datasets in
  the selected workspace. Rerunning the cell updates the same three examples.
- **An experiment row has an error:** open that row in LangSmith first. Confirm
  Bedrock model access, wallet balance, and WalletHub signing permission.
- **A judge score is missing:** confirm the configured Bedrock model supports
  structured output and inspect the evaluator error in the experiment.
- **A groundedness score fails:** compare the final answer with the captured
  HTTP evidence and trace. A model judge can be wrong; use its reasoning as a
  finding to investigate, not an automatic verdict.
- **A policy score fails:** inspect the trace and current endpoint price. A
  changed endpoint is a real evaluation finding, not a reason to force a pass.
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
