# Deep Agents on AWS

A standalone notebook that walks attendees through the Deep Agents capabilities
the workshop abstract promises on AWS: Bedrock, Bedrock Knowledge Bases,
AgentCore Browser, AgentCore Gateway/MCP, AgentCore Code Interpreter, S3, and
the AWS-region LangSmith instance.

This folder is self-contained. The notebook imports only the local `tools.py`
module plus installed packages.

## The arc

| Part | What | AWS + LangChain |
|------|------|-----------------|
| 0 | Setup + verify | Bedrock + LangSmith trace |
| 1 | First agent, KB tool, and researcher delegation | Deep Agents harness + Bedrock KB |
| 2 | Pluggable backends: State, Filesystem, S3, Composite routing | S3 + EFS pattern |
| 3 | Managed Browser + Code Interpreter | AgentCore Browser + Code Interpreter |
| 4 | Federated order/ticket APIs + refund approval | AgentCore Gateway/MCP + HITL |
| 5 | Long-term memory, AGENTS.md, and skills | S3-backed Store |
| 6 | Evaluate with LLM-as-judge + deterministic checks | LangSmith + OpenEvals |
| 7 | Deploy readiness + optional hosted deploy | LangGraph dev + AWS LangSmith UI |
| 8 | Review production loop | LangSmith UI |

## Prerequisites

1. Bedrock model access enabled in `us-east-1` for Claude Haiku 4.5, Claude Sonnet
   4.6, and Titan Embed v2. Model access is account-level, not CloudFormation.
2. AWS credentials that can deploy the CDK stack in `us-east-1`.
3. An AWS-region LangSmith account and API key. Use
   `LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com`, not the UI URL.
4. A local `.env` created from `.env.example`. Keep real credentials out of git.
5. AWS CDK CLI installed for provisioning, for example `npm install -g aws-cdk`.

## Install

Use `uv` to create and manage the local virtual environment. `uv sync` creates
`.venv/` in this folder and installs the dependencies from `pyproject.toml`. You
do not need to activate the environment manually if you run commands with
`uv run`.

```bash
cp .env.example .env
uv sync --python 3.12
uv sync --extra cdk --python 3.12        # only needed to deploy cdk_preprovision.py
```

## Configure LangSmith

Create or sign in to an account on the AWS-region LangSmith instance:
`https://aws.smith.langchain.com`. In that AWS-region LangSmith account, create
an API key from settings, then update the LangSmith values in `.env`:

```bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com
LANGSMITH_API_KEY=<your-aws-region-langsmith-api-key>
LANGSMITH_PROJECT=deepagents-aws-tour
```

Use the API endpoint above, not the UI URL. A key from the default
`smith.langchain.com` instance will not authenticate against the AWS-region
instance.

If you cannot use `uv`, use `requirements.txt` with a standard virtual
environment. Keep this as a fallback; `uv` is the supported path for the
workshop.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Provision AWS resources

`cdk_preprovision.py` provisions the S3 bucket, Bedrock Knowledge Base, initial KB
ingestion, public Browser demo page, order/issue Lambdas, Cognito auth for Gateway,
Gateway invoke role, an attendee IAM policy, and a workshop-scoped IAM user for
optional hosted AWS LangSmith UI deployment runtime credentials. It uses the seed
docs in `data/` and the Browser demo page in `public_docs/`.

Run the deploy, then run the helper. The helper reads the CloudFormation outputs,
creates or reuses the AgentCore Gateway, registers the Lambda MCP targets, fetches
the Cognito client secret without printing it, and writes the required values to
`.env`.

```bash
cdk bootstrap aws://<account-id>/us-east-1   # first time only for the AWS account/region
cdk deploy
uv run python scripts/register_gateway.py --write-env .env
```

If you installed with `requirements.txt` instead of `uv`, keep the virtual
environment activated and run:

```bash
cdk deploy --app "python cdk_preprovision.py"
python scripts/register_gateway.py --write-env .env
```

Attach the emitted `AttendeePolicyArn` to the AWS identity that will run the
notebook. That policy grants access to the provisioned S3 bucket, the Bedrock
Knowledge Base, Bedrock model invocation, AgentCore Browser/Code Interpreter, and
AgentCore Gateway APIs used by the tour.

## Run the notebook

```bash
uv run jupyter lab deepagents_aws_tour.ipynb
```

If you used the `requirements.txt` fallback, keep that environment activated and
run:

```bash
jupyter lab deepagents_aws_tour.ipynb
```

## Deploy access

Part 7 validates the deployable Python Deep Agent exported by `graph.py`, including
Gateway/MCP order and ticket tools, HITL-gated refunds, `query_product_kb`,
AgentCore Browser research, and S3-backed `/memories/` through the custom store hook
in `langgraph.json`.

For the AWS LangSmith tenant, the required workshop path is local deploy-readiness
validation. The `langgraph deploy` CLI path for the AWS tenant is coming soon; for
this workshop, use local validation as the required path and the AWS LangSmith UI as
the optional hosted-deploy path.

Required local validation:

```bash
export LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com
aws sts get-caller-identity
uv run python scripts/register_gateway.py --write-env .env
uv run langgraph validate
uv run python -c "import graph; print('graph import ok:', type(graph.graph).__name__)"
uv run langgraph dev --no-browser --port 2024
```

In a second terminal:

```bash
curl -sS http://127.0.0.1:2024/ok
```

Expected:

```json
{"ok":true}
```

If `aws sts get-caller-identity` returns `ExpiredToken`, refresh the same AWS
credentials you used for CDK and rerun the helper command.

Optional hosted deployment is covered at the bottom of Part 7 and in
`deploy/README.md`. It requires AWS LangSmith Deployment access, a repo visible to
the LangSmith GitHub app, and hosted runtime credentials generated with
`uv run python scripts/create_deployment_user_key.py --write-env .env`. When
deploying from `langchain-aws-samples`, select `examples/deepagents-aws-tour` as
the app directory in the UI. Without that setup, use local validation only or a
facilitator demo.

EFS is a pattern note only unless the attendee environment can mount EFS. S3
remains the hands-on durable backend.
