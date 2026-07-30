"""Build the one-time AgentCore Payments setup notebook.

The setup is adapted from the AWS AgentCore Payments tutorial identified in
NOTICE. It intentionally supports only Coinbase CDP on Base Sepolia.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK_PATH = HERE / "setup_agentcore_payments.ipynb"


def _source(text: str) -> list[str]:
    return textwrap.dedent(text).strip().splitlines(keepends=True)


def _markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


def _code(text: str) -> dict:
    source = textwrap.dedent(text).strip()
    compile(source, "<setup-notebook-cell>", "exec")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    _markdown(
        """
        # One-time setup: AgentCore Payments

        Create the AgentCore Payments resources used by
        `agentcore_payments.ipynb` without cloning another repository.

        This minimal path supports **Coinbase CDP**, **Base Sepolia**, and
        testnet USDC only. It creates four IAM roles, a credential provider,
        Payment Manager, connector, and embedded wallet instrument.

        > Testnet USDC has no real-world value. The AWS resources created here
        > can incur normal AWS charges until cleaned up.
        """
    ),
    _markdown(
        """
        ## Before you begin

        You need:

        - AWS credentials with AgentCore Payments preview access.
        - Permission to create and assume IAM roles, pass the resource role,
          and create AgentCore Payments and Secrets Manager resources.
        - A Coinbase account and Coinbase Developer Platform project.

        This notebook does not print Coinbase credentials or temporary AWS
        credentials. Keep the local `.env` file out of source control.
        """
    ),
    _markdown(
        """
        ## 1. Get three Coinbase CDP credentials

        In [Coinbase Developer Platform](https://portal.cdp.coinbase.com/):

        1. Use the project the portal opens. Coinbase may create a default
           project automatically; if you see a project switcher, create or
           select a project there.
        2. Open **API Keys → Secret API keys**. Do **not** choose
           **Client API Key**; AgentCore needs a server-side secret key.
        3. Create a Secret API Key. If asked for a signing algorithm, keep the
           default **Ed25519** option. Retain its API Key ID and API Key Secret.
        4. Open **Wallets → ServerWallet** and retain the Wallet Secret. It may
           be shown only once.
        5. Open **Wallets → Embedded Wallet → Policies** and enable
           **Delegated Signing**.

        Store the secrets in a password manager. For this local tutorial, add
        them only to `.env` using the names shown in the next section.
        """
    ),
    _markdown(
        """
        ## 2. Configure the local environment

        Copy `.env.example` to `.env` and fill in:

        - `COINBASE_API_KEY_ID`
        - `COINBASE_API_KEY_SECRET`
        - `COINBASE_WALLET_SECRET`
        - `LINKED_EMAIL` — an email you can use to sign in to WalletHub

        Leave `PAYMENT_MANAGER_ARN` and `INSTRUMENT_ID` blank. This notebook
        generates them and writes them back to the same `.env` file.
        """
    ),
    _code(
        """
        import os
        import time
        import uuid

        import boto3
        from dotenv import load_dotenv

        from setup_utils import (
            CONTROL_PLANE_ROLE,
            MANAGEMENT_ROLE,
            PROCESS_PAYMENT_ROLE,
            RESOURCE_RETRIEVAL_ROLE,
            assume_role,
            client_token,
            require_env,
            setup_payment_roles,
            wait_for_status,
            write_env,
        )

        load_dotenv(override=True)

        REGION = os.environ.get("AWS_REGION", "us-west-2")
        NETWORK = os.environ.get("NETWORK", "ETHEREUM").upper()
        if NETWORK != "ETHEREUM":
            raise ValueError(
                "This minimal setup supports ETHEREUM/Base Sepolia only."
            )

        USER_ID = os.environ.get("USER_ID", "test-user-001").strip()
        LINKED_EMAIL = require_env("LINKED_EMAIL")
        if "@" not in LINKED_EMAIL:
            raise ValueError("LINKED_EMAIL must be a valid email address.")

        COINBASE_API_KEY_ID = require_env("COINBASE_API_KEY_ID")
        COINBASE_API_KEY_SECRET = require_env("COINBASE_API_KEY_SECRET")
        COINBASE_WALLET_SECRET = require_env("COINBASE_WALLET_SECRET")

        base_session = boto3.Session(region_name=REGION)
        base_session.client("sts").get_caller_identity()
        write_env(
            {
                "AWS_REGION": REGION,
                "NETWORK": "ETHEREUM",
                "USER_ID": USER_ID,
            }
        )

        print("AWS credentials and required configuration verified.")
        print("Region:", REGION)
        print("Network: Base Sepolia testnet")
        """
    ),
    _markdown(
        """
        ## 3. Create the IAM roles

        The setup preserves AWS's role separation:

        | Role | Purpose |
        |---|---|
        | Control plane | Creates managers, connectors, and credential providers |
        | Management | Creates instruments and sessions; cannot process payments |
        | Process payment | Processes payments and reads runtime state |
        | Resource retrieval | Lets AgentCore retrieve provider credentials |

        Existing roles with these example-specific names are updated and
        reused.
        """
    ),
    _code(
        """
        roles = setup_payment_roles(REGION)
        CONTROL_PLANE_ROLE_ARN = roles["control_plane"]
        MANAGEMENT_ROLE_ARN = roles["management"]
        PROCESS_PAYMENT_ROLE_ARN = roles["process_payment"]
        RESOURCE_RETRIEVAL_ROLE_ARN = roles["resource_retrieval"]

        write_env(
            {
                "CONTROL_PLANE_ROLE_ARN": CONTROL_PLANE_ROLE_ARN,
                "MANAGEMENT_ROLE_ARN": MANAGEMENT_ROLE_ARN,
                "PROCESS_PAYMENT_ROLE_ARN": PROCESS_PAYMENT_ROLE_ARN,
                "RESOURCE_RETRIEVAL_ROLE_ARN": (
                    RESOURCE_RETRIEVAL_ROLE_ARN
                ),
            }
        )
        """
    ),
    _markdown(
        """
        ## 4. Create scoped setup clients

        Control-plane operations run through the control-plane role. Wallet
        and instrument operations run through the management role. Temporary
        credentials remain in memory and are never printed or written to
        `.env`.
        """
    ),
    _code(
        """
        control_session = assume_role(
            base_session,
            CONTROL_PLANE_ROLE_ARN,
            "langchain-agentcore-payments-control",
        )
        management_session = assume_role(
            base_session,
            MANAGEMENT_ROLE_ARN,
            "langchain-agentcore-payments-management",
        )

        control_endpoint = (
            f"https://bedrock-agentcore-control.{REGION}.amazonaws.com"
        )
        data_endpoint = f"https://bedrock-agentcore.{REGION}.amazonaws.com"

        control_client = control_session.client(
            "bedrock-agentcore-control",
            endpoint_url=control_endpoint,
        )
        data_client = management_session.client(
            "bedrock-agentcore",
            endpoint_url=data_endpoint,
        )
        print("Scoped AgentCore clients ready.")
        """
    ),
    _markdown(
        """
        ## 5. Create the Coinbase credential provider

        AgentCore stores the Coinbase credentials in its managed identity
        integration. The secrets are passed directly to the API and are not
        included in notebook output.
        """
    ),
    _code(
        """
        CREDENTIAL_PROVIDER_ARN = os.environ.get(
            "CREDENTIAL_PROVIDER_ARN",
            "",
        ).strip()
        CREDENTIAL_PROVIDER_NAME = os.environ.get(
            "CREDENTIAL_PROVIDER_NAME",
            "",
        ).strip()

        if bool(CREDENTIAL_PROVIDER_ARN) != bool(CREDENTIAL_PROVIDER_NAME):
            raise ValueError(
                "CREDENTIAL_PROVIDER_ARN and CREDENTIAL_PROVIDER_NAME must "
                "either both be set or both be blank."
            )

        if CREDENTIAL_PROVIDER_ARN:
            print("Reusing credential provider recorded in .env.")
        else:
            CREDENTIAL_PROVIDER_NAME = (
                f"LangChainCoinbase{uuid.uuid4().hex[:8]}"
            )
            response = control_client.create_payment_credential_provider(
                name=CREDENTIAL_PROVIDER_NAME,
                credentialProviderVendor="CoinbaseCDP",
                providerConfigurationInput={
                    "coinbaseCdpConfiguration": {
                        "apiKeyId": COINBASE_API_KEY_ID,
                        "apiKeySecret": COINBASE_API_KEY_SECRET,
                        "walletSecret": COINBASE_WALLET_SECRET,
                    }
                },
            )
            CREDENTIAL_PROVIDER_ARN = response["credentialProviderArn"]
            write_env(
                {
                    "CREDENTIAL_PROVIDER_NAME": CREDENTIAL_PROVIDER_NAME,
                    "CREDENTIAL_PROVIDER_ARN": CREDENTIAL_PROVIDER_ARN,
                }
            )
            print("Coinbase credential provider created.")
        """
    ),
    _markdown(
        """
        ## 6. Create the Payment Manager

        The Payment Manager is the top-level application resource. It uses the
        resource-retrieval role to access the Coinbase credentials at runtime.
        """
    ),
    _code(
        """
        PAYMENT_MANAGER_ID = os.environ.get(
            "PAYMENT_MANAGER_ID",
            "",
        ).strip()
        PAYMENT_MANAGER_ARN = os.environ.get(
            "PAYMENT_MANAGER_ARN",
            "",
        ).strip()

        if bool(PAYMENT_MANAGER_ID) != bool(PAYMENT_MANAGER_ARN):
            raise ValueError(
                "PAYMENT_MANAGER_ID and PAYMENT_MANAGER_ARN must either both "
                "be set or both be blank."
            )

        if PAYMENT_MANAGER_ID:
            print("Reusing Payment Manager recorded in .env.")
        else:
            manager_name = (
                f"LangChainAgentCorePayments{uuid.uuid4().hex[:8]}"
            )
            response = control_client.create_payment_manager(
                name=manager_name,
                description="LangChain AgentCore Payments example",
                authorizerType="AWS_IAM",
                roleArn=RESOURCE_RETRIEVAL_ROLE_ARN,
                clientToken=client_token(),
            )
            PAYMENT_MANAGER_ID = response["paymentManagerId"]
            PAYMENT_MANAGER_ARN = response["paymentManagerArn"]
            write_env(
                {
                    "PAYMENT_MANAGER_ID": PAYMENT_MANAGER_ID,
                    "PAYMENT_MANAGER_ARN": PAYMENT_MANAGER_ARN,
                }
            )

        wait_for_status(
            control_client.get_payment_manager,
            "READY",
            paymentManagerId=PAYMENT_MANAGER_ID,
        )
        print("Payment Manager is ready.")
        """
    ),
    _markdown(
        """
        ## 7. Connect the Payment Manager to Coinbase
        """
    ),
    _code(
        """
        PAYMENT_CONNECTOR_ID = os.environ.get(
            "PAYMENT_CONNECTOR_ID",
            "",
        ).strip()

        if PAYMENT_CONNECTOR_ID:
            print("Reusing payment connector recorded in .env.")
        else:
            response = control_client.create_payment_connector(
                paymentManagerId=PAYMENT_MANAGER_ID,
                name="LangChainCoinbaseConnector",
                description="Coinbase connector for the LangChain example",
                type="CoinbaseCDP",
                credentialProviderConfigurations=[
                    {
                        "coinbaseCDP": {
                            "credentialProviderArn": (
                                CREDENTIAL_PROVIDER_ARN
                            )
                        }
                    }
                ],
                clientToken=client_token(),
            )
            PAYMENT_CONNECTOR_ID = response["paymentConnectorId"]
            write_env(
                {"PAYMENT_CONNECTOR_ID": PAYMENT_CONNECTOR_ID}
            )

        wait_for_status(
            control_client.get_payment_connector,
            "READY",
            paymentManagerId=PAYMENT_MANAGER_ID,
            paymentConnectorId=PAYMENT_CONNECTOR_ID,
        )
        print("Coinbase connector is ready.")
        """
    ),
    _markdown(
        """
        ## 8. Create the embedded Base Sepolia wallet

        The payment instrument is an embedded wallet linked to `LINKED_EMAIL`
        and the application-level `USER_ID`. It is not a personal wallet
        private key and the model never receives its identifier.
        """
    ),
    _code(
        """
        INSTRUMENT_ID = os.environ.get("INSTRUMENT_ID", "").strip()
        WALLET_ADDRESS = os.environ.get("WALLET_ADDRESS", "").strip()

        if INSTRUMENT_ID:
            instrument_response = data_client.get_payment_instrument(
                paymentManagerArn=PAYMENT_MANAGER_ARN,
                paymentConnectorId=PAYMENT_CONNECTOR_ID,
                paymentInstrumentId=INSTRUMENT_ID,
                userId=USER_ID,
            )
            instrument = instrument_response["paymentInstrument"]
            if not WALLET_ADDRESS:
                WALLET_ADDRESS = instrument[
                    "paymentInstrumentDetails"
                ]["embeddedCryptoWallet"]["walletAddress"]
                write_env({"WALLET_ADDRESS": WALLET_ADDRESS})
            print("Reusing payment instrument recorded in .env.")
        else:
            instrument_response = data_client.create_payment_instrument(
                paymentManagerArn=PAYMENT_MANAGER_ARN,
                paymentConnectorId=PAYMENT_CONNECTOR_ID,
                userId=USER_ID,
                paymentInstrumentType="EMBEDDED_CRYPTO_WALLET",
                paymentInstrumentDetails={
                    "embeddedCryptoWallet": {
                        "network": "ETHEREUM",
                        "linkedAccounts": [
                            {
                                "email": {
                                    "emailAddress": LINKED_EMAIL
                                }
                            }
                        ],
                    }
                },
                clientToken=client_token(),
            )
            instrument = instrument_response["paymentInstrument"]
            INSTRUMENT_ID = instrument["paymentInstrumentId"]
            WALLET_ADDRESS = instrument[
                "paymentInstrumentDetails"
            ]["embeddedCryptoWallet"]["walletAddress"]
            write_env(
                {
                    "INSTRUMENT_ID": INSTRUMENT_ID,
                    "WALLET_ADDRESS": WALLET_ADDRESS,
                }
            )

        wait_for_status(
            data_client.get_payment_instrument,
            "ACTIVE",
            paymentManagerArn=PAYMENT_MANAGER_ARN,
            paymentConnectorId=PAYMENT_CONNECTOR_ID,
            paymentInstrumentId=INSTRUMENT_ID,
            userId=USER_ID,
        )
        print("Embedded wallet instrument is active.")
        """
    ),
    _markdown(
        """
        ## 9. Fund the wallet and grant signing permission

        These are the only unavoidable browser actions:

        1. Copy the wallet address printed by the next cell.
        2. Open [Circle Faucet](https://faucet.circle.com/), select
           **Base Sepolia**, paste the address, and request testnet USDC.
        3. Open the WalletHub URL, sign in with `LINKED_EMAIL`, and grant the
           agent signing permission if prompted.

        Do not use WalletHub credit-card, bank, or on-ramp options for this
        tutorial; those can involve real assets. Use the faucet only.
        """
    ),
    _code(
        """
        from bedrock_agentcore.payments import PaymentManager

        setup_payment_manager = PaymentManager(
            payment_manager_arn=PAYMENT_MANAGER_ARN,
            region_name=REGION,
            boto3_session=management_session,
        )

        wallet_hub_url = None
        for attempt in range(6):
            details = setup_payment_manager.get_payment_instrument(
                user_id=USER_ID,
                payment_instrument_id=INSTRUMENT_ID,
            )
            wallet = details.get(
                "paymentInstrumentDetails",
                {},
            ).get("embeddedCryptoWallet", {})
            wallet_hub_url = wallet.get("redirectUrl")
            if wallet_hub_url:
                break
            if attempt < 5:
                time.sleep(5)

        print("Wallet address:", WALLET_ADDRESS)
        print("Circle faucet: https://faucet.circle.com/ (Base Sepolia)")
        if wallet_hub_url:
            print("WalletHub:", wallet_hub_url)
        else:
            print(
                "WalletHub URL is still provisioning. Re-run this cell in "
                "a minute."
            )

        print(
            "ACTION REQUIRED: fund with testnet USDC and grant signing "
            "permission before continuing."
        )
        """
    ),
    _markdown(
        """
        ## 10. Verify the testnet balance

        After funding, run this cell. A non-zero balance means the setup is
        ready for `agentcore_payments.ipynb`.
        """
    ),
    _code(
        """
        balance_response = data_client.get_payment_instrument_balance(
            paymentManagerArn=PAYMENT_MANAGER_ARN,
            paymentConnectorId=PAYMENT_CONNECTOR_ID,
            paymentInstrumentId=INSTRUMENT_ID,
            userId=USER_ID,
            chain="BASE_SEPOLIA",
            token="USDC",
        )
        token_balance = balance_response.get("tokenBalance", {})
        balance = int(token_balance.get("amount", "0")) / 1_000_000
        print(f"Base Sepolia balance: {balance:.2f} testnet USDC")

        if balance <= 0:
            raise RuntimeError(
                "The wallet is not funded yet. Complete Step 9 and rerun "
                "this cell."
            )

        required_outputs = {
            "PAYMENT_MANAGER_ARN": PAYMENT_MANAGER_ARN,
            "USER_ID": USER_ID,
            "INSTRUMENT_ID": INSTRUMENT_ID,
        }
        if not all(required_outputs.values()):
            raise RuntimeError("Required setup outputs are incomplete.")

        print("Setup complete. The payment agent notebook is ready.")
        print("Generated .env keys:", ", ".join(required_outputs))
        """
    ),
    _markdown(
        """
        ## Cleanup

        Payment sessions created by the agent notebook expire automatically.
        The resources created here persist.

        When finished, delete them in this order from the AgentCore and IAM
        consoles:

        1. Payment instrument
        2. Payment connector
        3. Payment Manager
        4. Payment credential provider and its managed secret
        5. The four `LangChainAgentCorePayments...` IAM roles

        Do not delete them until you have finished testing the agent notebook.
        """
    ),
    _markdown(
        """
        ## Source

        Adapted from the AWS AgentCore Payments setup tutorial. See `NOTICE`
        and `LICENSE-APACHE` in this directory.
        """
    ),
]

for index, cell in enumerate(CELLS, start=1):
    cell["id"] = f"setup-cell-{index:02d}"


def build_notebook() -> str:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the generated notebook differs from its source.",
    )
    args = parser.parse_args()

    expected = build_notebook()
    if args.check:
        if not NOTEBOOK_PATH.exists():
            raise SystemExit(f"Missing generated notebook: {NOTEBOOK_PATH}")
        if NOTEBOOK_PATH.read_text(encoding="utf-8") != expected:
            raise SystemExit(
                "Setup notebook is out of date. Run: "
                "python build_setup_notebook.py"
            )
        print("Setup notebook is up to date and all code cells compile.")
        return

    NOTEBOOK_PATH.write_text(expected, encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
