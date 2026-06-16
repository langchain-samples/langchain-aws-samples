# Deep Agents on AWS

A standalone notebook that walks attendees through the Deep Agents capabilities
the workshop abstract promises on AWS: Bedrock, Bedrock Knowledge Bases,
AgentCore Code Interpreter, S3, and the AWS-region LangSmith instance.

This folder is self-contained. The notebook imports only the local `tools.py`
module plus installed packages.

## The arc

| Part | What | AWS + LangChain |
|------|------|-----------------|
| 0 | Setup + verify | Bedrock + LangSmith trace |
| 1 | First agent, KB tool, and researcher delegation | Deep Agents harness + Bedrock KB |
| 2 | Pluggable backends: State, Filesystem, S3, Composite routing | S3 + EFS pattern |
| 3 | Code Interpreter as a sandbox backend | AgentCore Code Interpreter |
| 4 | Long-term memory, AGENTS.md, and skills | S3-backed Store |
| 5 | Evaluate from templates + deterministic checks | LangSmith + OpenEvals |
| 6 | One-command deploy surface | LangSmith Deployment / `langgraph deploy` |
| 7 | Review production loop | LangSmith UI |

## Prerequisites

1. Bedrock model access enabled in `us-east-1` for Claude Haiku 4.5, Claude Sonnet
   4.6, and Titan Embed v2. Model access is account-level, not CloudFormation.
2. AWS credentials with the policy output by `cdk_preprovision.py`.
3. LangSmith on the AWS-region instance. Use
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
uv sync
uv sync --extra cdk        # only needed to deploy cdk_preprovision.py
```

## Configure LangSmith

Create or sign in to an account on the AWS-region LangSmith instance:
`https://aws.smith.langchain.com`. Create an API key from your LangSmith
settings, then add it to `.env`:

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

Open the notebook from that environment:

```bash
uv run jupyter lab deepagents_aws_tour.ipynb
# or, from the activated requirements.txt environment:
jupyter lab deepagents_aws_tour.ipynb
```

## Provision AWS resources

`cdk_preprovision.py` provisions the S3 bucket, Bedrock Knowledge Base, initial KB
ingestion, and an attendee IAM policy. It uses the seed docs in `data/`.

```bash
cdk bootstrap aws://<account-id>/us-east-1   # first time only for the AWS account/region
cdk deploy
```

If you installed with `requirements.txt` instead of `uv`, keep the virtual
environment activated and run:

```bash
cdk deploy --app "python cdk_preprovision.py"
```

After deploy, copy the stack outputs into `.env`. `BedrockKbId` becomes
`BEDROCK_KB_ID`; `DataBucketName` becomes `AGENT_FILES_BUCKET`.

```bash
uv run python - <<'PY'
import boto3, re
from pathlib import Path

outputs = {
    x["OutputKey"]: x["OutputValue"]
    for s in boto3.client("cloudformation", region_name="us-east-1").describe_stacks(
        StackName="TourPreprovisionStack"
    )["Stacks"]
    for x in s.get("Outputs", [])
}
p = Path(".env")
t = p.read_text()
for k, v in (("BEDROCK_KB_ID", outputs["BedrockKbId"]), ("AGENT_FILES_BUCKET", outputs["DataBucketName"])):
    t = re.sub(rf"(?m)^{k}=.*$", f"{k}={v}", t) if re.search(rf"(?m)^{k}=", t) else t + f"{k}={v}\n"
p.write_text(t)
print("Updated BEDROCK_KB_ID and AGENT_FILES_BUCKET in .env")
PY
```

Attach the emitted `AttendeePolicyArn` to the AWS identity that will run the
notebook. That policy grants access to the provisioned S3 bucket, the Bedrock
Knowledge Base, Bedrock model invocation, and AgentCore runtime APIs used by the
tour.

## Run the notebook

```bash
uv run python build_tour.py
uv run jupyter lab deepagents_aws_tour.ipynb
```

Edit cell content in `build_tour.py`, keep `tour.md` in sync, and regenerate the
`.ipynb`.

## Deploy access

- Part 6 uses LangSmith Deployment through `langgraph deploy`. It deploys the same
  Python Deep Agent exported by `graph.py`, including `query_product_kb`, the
  researcher sub-agent, and S3-backed `/memories/` through the custom store hook
  in `langgraph.json`.
- Cloud deployment requires a LangSmith workspace with Deployment access.
  Without deployment access, use a facilitator-provisioned agent for the SDK
  invocation cell.
- `uv run langgraph validate` checks the `langgraph.json` import path.
  `uv run langgraph dev --no-browser` validates the graph locally without Docker.
  `uv run langgraph deploy --name deepagents-aws-tour` performs the hosted deploy.
  If Docker is not available and your workspace supports remote builds, use
  `uv run langgraph deploy --name deepagents-aws-tour --remote`. If deploy returns
  `403 Forbidden`, the key or workspace lacks Deployment access or points at the
  wrong LangSmith workspace.
- EFS is hands-on only if the attendee environment can mount EFS. Otherwise, the
  notebook uses a temp directory to teach the `FilesystemBackend` pattern, while
  S3 remains fully hands-on.

## Files not to ship

The `.gitignore` excludes local secrets and generated state: `.env`, `.venv/`,
`cdk.out/`, `.langgraph_api/`, `__pycache__/`, and `.ipynb_checkpoints/`. Keep
`.env.example`, `uv.lock`, `data/`, and the notebook in the published folder.
