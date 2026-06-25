"""S3-backed LangGraph store for LangSmith Deployment.

LangGraph API loads this factory from `langgraph.json` and exposes the yielded
store to graph runs. The Deep Agent's `/memories/` route uses `StoreBackend`
without a graph-level custom store, so local dev and hosted deployment both keep
S3-backed long-term memory without tripping the custom-store guard.
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

from tools import S3Store

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env", override=True)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is required for the S3-backed LangGraph store. "
            "Run scripts/register_gateway.py --write-env .env after CDK deploy."
        )
    return value


@asynccontextmanager
async def generate_store() -> AsyncIterator[S3Store]:
    """Yield the S3-backed store used by deployed `/memories/` files."""
    yield S3Store(bucket=_require_env("AGENT_FILES_BUCKET"), prefix="tour-memories")
