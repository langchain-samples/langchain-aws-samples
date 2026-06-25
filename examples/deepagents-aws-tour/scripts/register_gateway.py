"""Create/reuse AgentCore Gateway and register Lambda-backed MCP targets.

Run after `cdk deploy`:
    uv run python scripts/register_gateway.py --write-env .env

The Cognito client secret is fetched from Cognito and written only to the local
ignored .env file when --write-env is provided. Secret values are not printed.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = os.environ.get("AWS_REGION", "us-east-1")
STACK_NAME = os.environ.get("TOUR_STACK_NAME", "TourPreprovisionStack")
GATEWAY_NAME = os.environ.get("TOUR_GATEWAY_NAME", "deepagents-tour-gateway")
SECRET_KEYS = {"COGNITO_CLIENT_SECRET"}


ORDER_TOOLS = [
    {
        "name": "lookup_order",
        "description": "Look up an order by ID. Returns delivery status, items, total, and customer ID.",
        "inputSchema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": (
            "Issue a refund against an order. DESTRUCTIVE: the workshop agent gates "
            "this tool with human-in-the-loop approval before Gateway invocation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount_usd": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "amount_usd", "reason"],
        },
    },
]


ISSUE_TOOLS = [
    {
        "name": "lookup_customer_tickets",
        "description": "Return the customer's prior support tickets.",
        "inputSchema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    }
]


def _stack_outputs() -> dict[str, str]:
    cf = boto3.client("cloudformation", region_name=REGION)
    stack = cf.describe_stacks(StackName=STACK_NAME)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}


def _client_secret(user_pool_id: str, client_id: str) -> str:
    cognito = boto3.client("cognito-idp", region_name=REGION)
    return cognito.describe_user_pool_client(UserPoolId=user_pool_id, ClientId=client_id)["UserPoolClient"][
        "ClientSecret"
    ]


def _find_gateway_id(client, name: str) -> str | None:
    kwargs: dict = {}
    while True:
        resp = client.list_gateways(**kwargs)
        for gateway in resp.get("items", []):
            if gateway.get("name") == name:
                return gateway["gatewayId"]
        token = resp.get("nextToken")
        if not token:
            return None
        kwargs = {"nextToken": token}


def _expected_authorizer(discovery_url: str, client_id: str) -> dict:
    return {
        "customJWTAuthorizer": {
            "discoveryUrl": discovery_url,
            "allowedClients": [client_id],
        }
    }


def _wait_gateway_ready(client, gateway_id: str, *, label: str) -> None:
    for _ in range(60):
        gateway = client.get_gateway(gatewayIdentifier=gateway_id)
        status = gateway.get("status")
        if status == "READY":
            return
        if status in {"FAILED", "DELETING"}:
            raise RuntimeError(f"Gateway {label} ended in status {status}: {gateway.get('statusReasons')}")
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for Gateway {label} to become READY.")


def _wait_target_ready(client, gateway_id: str, target_id: str, *, label: str) -> None:
    for _ in range(60):
        target = client.get_gateway_target(gatewayIdentifier=gateway_id, targetId=target_id)
        status = target.get("status")
        if status == "READY":
            return
        if status in {"FAILED", "DELETING"}:
            raise RuntimeError(f"Gateway target {label} ended in status {status}: {target.get('statusReasons')}")
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for Gateway target {label} to become READY.")


def _existing_targets(client, gateway_id: str) -> dict[str, str]:
    targets: dict[str, str] = {}
    kwargs = {"gatewayIdentifier": gateway_id}
    while True:
        resp = client.list_gateway_targets(**kwargs)
        for target in resp.get("items", []):
            targets[target["name"]] = target["targetId"]
        token = resp.get("nextToken")
        if not token:
            return targets
        kwargs = {"gatewayIdentifier": gateway_id, "nextToken": token}


def _target_configuration(lambda_arn: str, tools: list[dict]) -> dict:
    return {
        "mcp": {
            "lambda": {
                "lambdaArn": lambda_arn,
                "toolSchema": {"inlinePayload": tools},
            }
        }
    }


def _target_lambda_arn(target: dict) -> str | None:
    return target.get("targetConfiguration", {}).get("mcp", {}).get("lambda", {}).get("lambdaArn")


def _ensure_target(client, gateway_id: str, existing: dict[str, str], name: str, lambda_arn: str, tools: list[dict]) -> None:
    target_config = _target_configuration(lambda_arn, tools)
    credentials = [{"credentialProviderType": "GATEWAY_IAM_ROLE"}]
    if name in existing:
        target_id = existing[name]
        target = client.get_gateway_target(gatewayIdentifier=gateway_id, targetId=target_id)
        if _target_lambda_arn(target) == lambda_arn:
            print(f"Target '{name}' already matches current stack, skipping.")
            return
        client.update_gateway_target(
            gatewayIdentifier=gateway_id,
            targetId=target_id,
            name=name,
            targetConfiguration=target_config,
            credentialProviderConfigurations=credentials,
        )
        print(f"Target '{name}' updated for current Lambda.")
        _wait_target_ready(client, gateway_id, target_id, label=name)
        return

    created = client.create_gateway_target(
        gatewayIdentifier=gateway_id,
        name=name,
        targetConfiguration=target_config,
        credentialProviderConfigurations=credentials,
    )
    print(f"Target '{name}' registered.")
    _wait_target_ready(client, gateway_id, created["targetId"], label=name)


def _gateway_mcp_url(gateway_url: str | None, gateway_id: str) -> str:
    base = (gateway_url or f"https://{gateway_id}.gateway.bedrock-agentcore.{REGION}.amazonaws.com").rstrip("/")
    return base if base.endswith("/mcp") else f"{base}/mcp"


def _upsert_env(path: Path, values: dict[str, str]) -> None:
    text = path.read_text() if path.exists() else ""
    for key, value in values.items():
        line = f"{key}={value}"
        if re.search(rf"(?m)^{re.escape(key)}=", text):
            text = re.sub(rf"(?m)^{re.escape(key)}=.*$", line, text)
        else:
            if text and not text.endswith("\n"):
                text += "\n"
            text += line + "\n"
    path.write_text(text)


def _print_values(values: dict[str, str], *, wrote_env: bool) -> None:
    print("\nGateway ready. Configured values:\n")
    for key, value in values.items():
        if key in SECRET_KEYS:
            suffix = "written to .env" if wrote_env else "hidden; rerun with --write-env .env"
            print(f"{key}=<{suffix}>")
        else:
            print(f"{key}={value}")


def register_gateway() -> dict[str, str]:
    outputs = _stack_outputs()
    user_pool_id = outputs["CognitoUserPoolId"]
    client_id = outputs["CognitoClientId"]
    discovery_url = f"https://cognito-idp.{REGION}.amazonaws.com/{user_pool_id}/.well-known/openid-configuration"

    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    expected_authorizer = _expected_authorizer(discovery_url, client_id)
    gateway_id = _find_gateway_id(control, GATEWAY_NAME)
    gateway_url = None
    if gateway_id:
        print(f"Reusing gateway '{GATEWAY_NAME}' ({gateway_id}).")
        gateway = control.get_gateway(gatewayIdentifier=gateway_id)
        current_authorizer = gateway.get("authorizerConfiguration")
        if current_authorizer != expected_authorizer or gateway.get("roleArn") != outputs["GatewayRoleArn"]:
            control.update_gateway(
                gatewayIdentifier=gateway_id,
                name=GATEWAY_NAME,
                roleArn=outputs["GatewayRoleArn"],
                protocolType="MCP",
                authorizerType="CUSTOM_JWT",
                authorizerConfiguration=expected_authorizer,
            )
            print("Gateway authorizer updated for current Cognito client.")
            _wait_gateway_ready(control, gateway_id, label=GATEWAY_NAME)
    else:
        gateway = control.create_gateway(
            name=GATEWAY_NAME,
            protocolType="MCP",
            authorizerType="CUSTOM_JWT",
            authorizerConfiguration=expected_authorizer,
            roleArn=outputs["GatewayRoleArn"],
        )
        gateway_id = gateway["gatewayId"]
        gateway_url = gateway.get("gatewayUrl")
        print(f"Created gateway '{GATEWAY_NAME}' ({gateway_id}).")
        _wait_gateway_ready(control, gateway_id, label=GATEWAY_NAME)

    existing = _existing_targets(control, gateway_id)
    _ensure_target(control, gateway_id, existing, "orders", outputs["OrderLambdaArn"], ORDER_TOOLS)
    _ensure_target(control, gateway_id, existing, "issues", outputs["IssueLambdaArn"], ISSUE_TOOLS)

    if not gateway_url:
        gateway_url = control.get_gateway(gatewayIdentifier=gateway_id).get("gatewayUrl")

    values = {
        "BEDROCK_KB_ID": outputs["BedrockKbId"],
        "AGENT_FILES_BUCKET": outputs["DataBucketName"],
        "PUBLIC_SUPPORT_DOC_KEY": outputs.get("PublicSupportDocKey", "public-docs/sh-hub-v2-troubleshooting.html"),
        "GATEWAY_URL": _gateway_mcp_url(gateway_url, gateway_id),
        "COGNITO_TOKEN_URL": outputs["CognitoTokenUrl"],
        "COGNITO_CLIENT_ID": client_id,
        "COGNITO_CLIENT_SECRET": _client_secret(user_pool_id, client_id),
    }
    optional_output_map = {
        "EfsFileSystemId": "EFS_FILE_SYSTEM_ID",
        "EfsAccessPointId": "EFS_ACCESS_POINT_ID",
    }
    for output_key, env_key in optional_output_map.items():
        if outputs.get(output_key):
            values[env_key] = outputs[output_key]
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-env", type=Path, help="Update a local .env file with CDK and Gateway values.")
    args = parser.parse_args()

    try:
        values = register_gateway()
    except NoCredentialsError as exc:
        raise SystemExit(
            "AWS credentials were not found. Configure or refresh the same AWS credentials "
            "you used for `cdk deploy`, verify with `aws sts get-caller-identity`, then rerun "
            "`uv run python scripts/register_gateway.py --write-env .env`."
        ) from exc
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"ExpiredToken", "ExpiredTokenException"}:
            raise SystemExit(
                "AWS credentials are expired. Refresh the same AWS credentials you used for "
                "`cdk deploy`, verify with `aws sts get-caller-identity`, then rerun "
                "`uv run python scripts/register_gateway.py --write-env .env`."
            ) from exc
        raise

    if args.write_env:
        _upsert_env(args.write_env, values)
        print(f"\nUpdated {args.write_env}")
    _print_values(values, wrote_env=bool(args.write_env))


if __name__ == "__main__":
    main()
