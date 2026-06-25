"""Create an access key for the hosted LangSmith Deployment IAM user.

Run after `cdk deploy`:
    uv run python scripts/create_deployment_user_key.py --write-env .env

The access key secret is written only to the local ignored .env file and is not
printed. Delete or rotate this key after the workshop.

To clean up after the workshop:
    uv run python scripts/create_deployment_user_key.py --delete-existing
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = os.environ.get("AWS_REGION", "us-east-1")
STACK_NAME = os.environ.get("TOUR_STACK_NAME", "TourPreprovisionStack")
SECRET_KEYS = {"AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}


def _stack_outputs() -> dict[str, str]:
    cf = boto3.client("cloudformation", region_name=REGION)
    stack = cf.describe_stacks(StackName=STACK_NAME)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}


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


def _print_values(values: dict[str, str], *, wrote_env: bool, user_name: str) -> None:
    print(f"Created hosted deployment access key for IAM user: {user_name}")
    print("\nConfigured values:\n")
    for key, value in values.items():
        if key in SECRET_KEYS:
            suffix = "written to .env" if wrote_env else "hidden; rerun with --write-env .env"
            print(f"{key}=<{suffix}>")
        else:
            print(f"{key}={value}")
    print("\nRotate or delete this access key after the workshop.")


def _deployment_user_name() -> str:
    outputs = _stack_outputs()
    return outputs["HostedDeploymentUserName"]


def _delete_existing_keys(iam, user_name: str) -> int:
    existing = iam.list_access_keys(UserName=user_name)["AccessKeyMetadata"]
    for item in existing:
        iam.delete_access_key(UserName=user_name, AccessKeyId=item["AccessKeyId"])
    return len(existing)


def delete_deployment_user_keys() -> None:
    user_name = _deployment_user_name()
    iam = boto3.client("iam")
    count = _delete_existing_keys(iam, user_name)
    print(f"Deleted {count} hosted deployment access key(s) for IAM user: {user_name}")


def create_deployment_user_key(*, rotate: bool = False) -> tuple[str, dict[str, str]]:
    user_name = _deployment_user_name()

    iam = boto3.client("iam")
    if rotate:
        deleted = _delete_existing_keys(iam, user_name)
        if deleted:
            print(f"Deleted {deleted} existing hosted deployment access key(s) before creating a new one.")

    existing = iam.list_access_keys(UserName=user_name)["AccessKeyMetadata"]
    if len(existing) >= 2:
        raise RuntimeError(
            f"IAM user {user_name} already has two access keys. Delete an old key in the IAM console, "
            "or rerun this script with --rotate."
        )
    if existing:
        print(
            f"IAM user {user_name} already has {len(existing)} access key(s). "
            "Creating one additional key. Use --rotate to replace existing keys instead."
        )

    key = iam.create_access_key(UserName=user_name)["AccessKey"]
    values = {
        "AWS_REGION": REGION,
        "AWS_DEFAULT_REGION": REGION,
        "AWS_ACCESS_KEY_ID": key["AccessKeyId"],
        "AWS_SECRET_ACCESS_KEY": key["SecretAccessKey"],
        # Clear any stale temporary session token. Empty values are not uploaded
        # as LangGraph deployment secrets.
        "AWS_SESSION_TOKEN": "",
    }
    return user_name, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-env",
        type=Path,
        help="Update a local .env file with hosted deployment AWS creds.",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Delete existing hosted deployment access keys before creating a new one.",
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing hosted deployment access keys and exit.",
    )
    args = parser.parse_args()

    try:
        if args.delete_existing:
            delete_deployment_user_keys()
            return
        if not args.write_env:
            parser.error("--write-env is required unless --delete-existing is used.")

        user_name, values = create_deployment_user_key(rotate=args.rotate)
    except NoCredentialsError as exc:
        raise SystemExit(
            "AWS credentials were not found. Configure or refresh the same AWS credentials "
            "you used for `cdk deploy`, verify with `aws sts get-caller-identity`, then rerun "
            "the hosted deployment key command."
        ) from exc
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"ExpiredToken", "ExpiredTokenException"}:
            raise SystemExit(
                "AWS credentials are expired. Refresh the same AWS credentials you used for "
                "`cdk deploy`, verify with `aws sts get-caller-identity`, then rerun the "
                "hosted deployment key command."
            ) from exc
        raise

    _upsert_env(args.write_env, values)
    print(f"\nUpdated {args.write_env}")
    _print_values(values, wrote_env=True, user_name=user_name)


if __name__ == "__main__":
    main()
