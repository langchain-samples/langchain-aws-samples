# LangSmith Deployment scaffold (Part 6)

Part 6 ships the same Python Deep Agent that the notebook builds. `langgraph.json`
points LangSmith Deployment at `graph.py:graph`, so the deployed graph keeps the
Bedrock Knowledge Base tool, researcher sub-agent, support-reply guidance, and
S3-backed `/memories/` through LangGraph's custom store hook.

```bash
# Run from the repo root, where langgraph.json lives.

# The CLI reads .env via langgraph.json. Export these too if your shell has stale
# values from another LangSmith workspace:
export LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com
export LANGSMITH_API_KEY=...              # reference your AWS-instance key, never paste it into files

uv run langgraph validate                 # check langgraph.json and graph import path
uv run langgraph dev --no-browser         # local API server, no Docker required
uv run langgraph deploy --name deepagents-aws-tour  # cloud deploy, local Docker build

# If Docker is not available and your workspace supports remote builds:
uv run langgraph deploy --name deepagents-aws-tour --remote
```

Gotchas (all hit during testing):
- **Run from the directory with `langgraph.json`**.
- **Cloud deployment requires LangSmith Deployment access**. Docker is only required for local builds when remote build is not available.
- **Use the AWS-region LangSmith endpoint**. A stale `LANGSMITH_ENDPOINT` pointing at the default instance deploys to the wrong workspace.
- **LangSmith Deployment plan access is workspace-specific**. If attendees do not have deployment access, use a facilitator-provisioned `DEPLOYMENT_URL`.
- **Skipped reserved `LANGSMITH_*` variables are expected**. The CLI reads them for deploy, but does not add them as deployment secrets.
- **`403 Forbidden` is an access problem**. The key or workspace lacks Deployment access, or it points at the wrong LangSmith workspace.
- **Custom deployment stores are alpha**. The S3 store is registered in `langgraph.json` instead of passed as `create_deep_agent(..., store=...)`.
