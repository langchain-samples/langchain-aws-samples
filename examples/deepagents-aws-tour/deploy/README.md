# Part 7 deploy readiness

Part 7 validates the same Python Deep Agent that the notebook builds. `langgraph.json`
points at `graph.py:graph`, so the deployable graph keeps Gateway/MCP order and ticket
tools, HITL-gated refunds, the Bedrock Knowledge Base tool, AgentCore Browser research,
support-reply guidance, and S3-backed `/memories/` through LangGraph's custom store
hook.

For the AWS LangSmith tenant, the required workshop path is local deploy-readiness
validation. The `langgraph deploy` CLI path for the AWS tenant is coming soon; for
this workshop, use local validation as the required path and the AWS LangSmith UI as
the optional hosted-deploy path.

## Required local validation

Run from the sample directory, `examples/deepagents-aws-tour`, where
`langgraph.json` lives:

```bash
export LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com

# Confirm your current shell AWS credentials can still see the CDK stack. If this
# returns ExpiredToken, refresh your AWS credentials before continuing.
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

The required workshop path ends here. Stop the local server with `Ctrl-C` when you
are done testing.

## Required-path gotchas

- **Run from the directory with `langgraph.json`** for local validation. In
  `langchain-aws-samples`, that directory is `examples/deepagents-aws-tour`.
- **AWS helpers use your current shell AWS credentials**. If `aws sts get-caller-identity` or a helper returns `ExpiredToken`, refresh the same credentials you used for CDK and rerun the command.
- **Gateway, KB, and S3 env vars are required at import time**. Run `uv run python scripts/register_gateway.py --write-env .env` after `cdk deploy` before the import smoke test or `langgraph dev`.
- **Custom deployment stores are alpha**. The S3 store is registered in `langgraph.json` instead of passed as `create_deep_agent(..., store=...)`.

## Optional hosted deploy in AWS LangSmith

Use this path only when the attendee or facilitator has:

- AWS LangSmith Deployment access.
- A GitHub account.
- A fork or repo that the AWS LangSmith GitHub app can access.
- Stack-derived runtime env vars from `.env`.

Fork the workshop repo. If you are using `langchain-aws-samples`, fork that repo
and deploy from the branch that contains `examples/deepagents-aws-tour`:

```bash
# Option A: GitHub CLI
gh repo fork <workshop-repo-owner>/<workshop-repo-name> --clone=false

# Option B: browser
# Open langchain-aws-samples on GitHub, click Fork, and create a fork under your account or org.
```

If you already have the repo locally and need to push a branch to your fork:

```bash
git remote add fork git@github.com:<your-github-user-or-org>/<workshop-repo-name>.git
git push fork <branch-name>
```

Create or update hosted runtime AWS credentials:

```bash
aws sts get-caller-identity
uv run python scripts/create_deployment_user_key.py --write-env .env
```

In `https://aws.smith.langchain.com`:

1. Open **Deployments**.
2. Create a new deployment.
3. Connect GitHub if prompted.
4. Select the fork and branch.
5. Use `examples/deepagents-aws-tour` as the app directory.
6. Use `langgraph.json` as the config, relative to that app directory.
7. Confirm the graph id is `support_tour`.
8. Add the required env vars/secrets from `.env`.
9. Deploy.
10. Copy the deployment API URL.

Required deployment env vars/secrets:

```text
AWS_REGION
AWS_DEFAULT_REGION
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
BEDROCK_KB_ID
AGENT_FILES_BUCKET
PUBLIC_SUPPORT_DOC_KEY
GATEWAY_URL
COGNITO_TOKEN_URL
COGNITO_CLIENT_ID
COGNITO_CLIENT_SECRET
MCP_TRANSPORT
```

Do not commit `.env`, and do not add these local-only values to the deployment:

```text
DEPLOYMENT_URL
DEPLOYMENT_GRAPH
LANGSMITH_DEPLOYMENT_NAME
LANGGRAPH_HOST_URL
```

After the UI deploy succeeds:

```bash
export DEPLOYMENT_URL=<deployment-api-url>
export DEPLOYMENT_GRAPH=support_tour
```

Then run the optional Part 7 SDK invocation and feedback cells.

Hosted deploy notes:

- **Hosted AWS runtime credentials are required**. Use `uv run python scripts/create_deployment_user_key.py --rotate --write-env .env` to replace stale hosted credentials, then update the deployment env/secrets in the UI.
- **GitHub setup is workspace/account-specific**. If the LangSmith GitHub app cannot see the repo, fork it into the connected account or org.
- **The AWS tenant uses the UI path for now**. `langgraph deploy` for the AWS tenant is coming soon; use the AWS LangSmith UI path for this workshop.
