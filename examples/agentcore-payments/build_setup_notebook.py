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
        3. Create a Secret API Key with nickname
           `agentcore-payments-tutorial`. Select **Opt-out of IP allowlisting**
           because AgentCore calls Coinbase from AWS; a private laptop address
           such as `192.168...` will not work. If your organization requires
           an allowlist, ask the AgentCore Payments team for supported outbound
           IPs. Keep the default **Ed25519** signing algorithm.
        4. Retain the API Key ID and API Key Secret when they are shown.
        5. Open
           [**Wallets → Non-custodial Wallet → Security**](https://portal.cdp.coinbase.com/wallets/non-custodial/security).
           Do not choose Custodial Wallet or Agentic Wallet.
        6. Under **Wallet Secret**, generate and retain the secret. It may be
           shown only once.
        7. On the same **Security** page, enable **Delegated Signing**.

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

        Do not run cleanup until you have finished the agent notebook. This is
        destructive: the embedded wallet instrument and AgentCore-managed
        Coinbase credentials cannot be recovered from AgentCore afterward.

        The next cell is self-contained and safe to rerun. It reads only the
        recorded resource identifiers from `.env`, prints whether each target
        is recorded without printing its value, and requires an exact
        confirmation phrase before deleting anything.

        It removes AWS resources in dependency order:

        1. Payment sessions created by the workshop
        2. Embedded wallet instrument
        3. Coinbase payment connector
        4. Payment Manager
        5. Coinbase credential provider and its AgentCore-managed secret
        6. The four `LangChainAgentCorePayments...` IAM roles
        7. Generated resource identifiers from the local `.env`

        If cleanup stops, fix the reported dependency or permission issue and
        rerun the same cell. Values are cleared from `.env` only after every
        AWS deletion succeeds.
        """
    ),
    _code(
        """
        import os

        import boto3
        from botocore.exceptions import ClientError
        from dotenv import load_dotenv

        from setup_utils import (
            CONTROL_PLANE_ROLE,
            MANAGEMENT_ROLE,
            assume_role,
            client_token,
            delete_payment_roles,
            is_not_found,
            setup_payment_roles,
            wait_for_deleted,
            write_env,
        )

        load_dotenv(override=True)

        EXPECTED_CLEANUP_CONFIRMATION = (
            "DELETE AGENTCORE PAYMENTS TEST RESOURCES"
        )
        CLEANUP_CONFIRMATION = ""

        cleanup_keys = [
            "PAYMENT_MANAGER_ARN",
            "PAYMENT_MANAGER_ID",
            "PAYMENT_CONNECTOR_ID",
            "CREDENTIAL_PROVIDER_NAME",
            "CREDENTIAL_PROVIDER_ARN",
            "INSTRUMENT_ID",
            "WALLET_ADDRESS",
            "CONTROL_PLANE_ROLE_ARN",
            "MANAGEMENT_ROLE_ARN",
            "PROCESS_PAYMENT_ROLE_ARN",
            "RESOURCE_RETRIEVAL_ROLE_ARN",
        ]
        cleanup_values = {
            key: os.environ.get(key, "").strip()
            for key in cleanup_keys
        }

        print("AWS cleanup targets from .env:")
        for key in cleanup_keys:
            state = "recorded" if cleanup_values[key] else "not recorded"
            print(f"- {key}: {state}")

        if CLEANUP_CONFIRMATION != EXPECTED_CLEANUP_CONFIRMATION:
            raise RuntimeError(
                "Cleanup is locked. Set CLEANUP_CONFIRMATION to "
                f"{EXPECTED_CLEANUP_CONFIRMATION!r} in this cell and rerun "
                "it only after finishing the workshop."
            )

        cleanup_region = os.environ.get("AWS_REGION", "us-west-2")
        cleanup_user_id = os.environ.get(
            "USER_ID",
            "test-user-001",
        ).strip()
        manager_arn = cleanup_values["PAYMENT_MANAGER_ARN"]
        manager_id = cleanup_values["PAYMENT_MANAGER_ID"]
        connector_id = cleanup_values["PAYMENT_CONNECTOR_ID"]
        provider_name = cleanup_values["CREDENTIAL_PROVIDER_NAME"]
        provider_arn = cleanup_values["CREDENTIAL_PROVIDER_ARN"]
        instrument_id = cleanup_values["INSTRUMENT_ID"]

        if bool(manager_arn) != bool(manager_id):
            raise RuntimeError(
                "PAYMENT_MANAGER_ARN and PAYMENT_MANAGER_ID must both be "
                "recorded or both be blank before cleanup."
            )
        if connector_id and not manager_id:
            raise RuntimeError(
                "PAYMENT_CONNECTOR_ID is recorded without its Payment "
                "Manager. Restore the manager values in .env or delete the "
                "connector in the AgentCore console."
            )
        if instrument_id and not (manager_arn and connector_id):
            raise RuntimeError(
                "INSTRUMENT_ID is recorded without its manager and connector. "
                "Restore those values in .env or delete the instrument in "
                "the AgentCore console."
            )
        if provider_arn and not provider_name:
            raise RuntimeError(
                "CREDENTIAL_PROVIDER_ARN is recorded without "
                "CREDENTIAL_PROVIDER_NAME. Restore the provider name in "
                ".env or delete the credential provider in the AgentCore "
                "console."
            )

        core_resources_recorded = any(
            [
                manager_arn,
                manager_id,
                connector_id,
                provider_name,
                instrument_id,
            ]
        )

        if core_resources_recorded:
            cleanup_base_session = boto3.Session(
                region_name=cleanup_region
            )
            cleanup_base_session.client("sts").get_caller_identity()

            # Update or recreate the tutorial roles so this cleanup cell has
            # the current DeletePaymentSession permission.
            cleanup_roles = setup_payment_roles(cleanup_region)

            def cleanup_session(role_arn: str, session_name: str):
                try:
                    return assume_role(
                        cleanup_base_session,
                        role_arn,
                        session_name,
                    )
                except ClientError as error:
                    code = error.response.get("Error", {}).get("Code", "")
                    if code in {"AccessDenied", "NoSuchEntity"}:
                        print(
                            "Could not assume a tutorial cleanup role; using "
                            "the current AWS credentials instead."
                        )
                        return cleanup_base_session
                    raise

            cleanup_control_session = cleanup_session(
                cleanup_roles["control_plane"],
                "langchain-payments-cleanup-control",
            )
            cleanup_management_session = cleanup_session(
                cleanup_roles["management"],
                "langchain-payments-cleanup-management",
            )
            cleanup_control_client = cleanup_control_session.client(
                "bedrock-agentcore-control",
                endpoint_url=(
                    "https://bedrock-agentcore-control."
                    f"{cleanup_region}.amazonaws.com"
                ),
            )
            cleanup_data_client = cleanup_management_session.client(
                "bedrock-agentcore",
                endpoint_url=(
                    f"https://bedrock-agentcore.{cleanup_region}.amazonaws.com"
                ),
            )

            def delete_or_skip(delete_call, label: str, **kwargs) -> bool:
                try:
                    delete_call(**kwargs)
                    print(f"{label}: delete requested")
                    return True
                except ClientError as error:
                    if is_not_found(error):
                        print(f"{label}: already absent")
                        return False
                    raise

            if manager_arn:
                session_ids = []
                next_token = None
                while True:
                    list_kwargs = {
                        "paymentManagerArn": manager_arn,
                        "userId": cleanup_user_id,
                        "maxResults": 100,
                    }
                    if next_token:
                        list_kwargs["nextToken"] = next_token
                    try:
                        response = (
                            cleanup_data_client.list_payment_sessions(
                                **list_kwargs
                            )
                        )
                    except ClientError as error:
                        if is_not_found(error):
                            break
                        raise
                    session_ids.extend(
                        session["paymentSessionId"]
                        for session in response.get("paymentSessions", [])
                        if session.get("paymentSessionId")
                    )
                    next_token = response.get("nextToken")
                    if not next_token:
                        break

                print("Payment sessions found:", len(session_ids))
                for payment_session_id in session_ids:
                    requested = delete_or_skip(
                        cleanup_data_client.delete_payment_session,
                        "Payment session",
                        paymentManagerArn=manager_arn,
                        paymentSessionId=payment_session_id,
                        userId=cleanup_user_id,
                    )
                    if requested:
                        wait_for_deleted(
                            cleanup_data_client.get_payment_session,
                            "Payment session",
                            paymentManagerArn=manager_arn,
                            paymentSessionId=payment_session_id,
                            userId=cleanup_user_id,
                        )

            if instrument_id:
                requested = delete_or_skip(
                    cleanup_data_client.delete_payment_instrument,
                    "Payment instrument",
                    paymentManagerArn=manager_arn,
                    paymentConnectorId=connector_id,
                    paymentInstrumentId=instrument_id,
                    userId=cleanup_user_id,
                )
                if requested:
                    wait_for_deleted(
                        cleanup_data_client.get_payment_instrument,
                        "Payment instrument",
                        paymentManagerArn=manager_arn,
                        paymentConnectorId=connector_id,
                        paymentInstrumentId=instrument_id,
                        userId=cleanup_user_id,
                    )

            if connector_id:
                requested = delete_or_skip(
                    cleanup_control_client.delete_payment_connector,
                    "Payment connector",
                    paymentManagerId=manager_id,
                    paymentConnectorId=connector_id,
                    clientToken=client_token(),
                )
                if requested:
                    wait_for_deleted(
                        cleanup_control_client.get_payment_connector,
                        "Payment connector",
                        paymentManagerId=manager_id,
                        paymentConnectorId=connector_id,
                    )

            if manager_id:
                requested = delete_or_skip(
                    cleanup_control_client.delete_payment_manager,
                    "Payment Manager",
                    paymentManagerId=manager_id,
                    clientToken=client_token(),
                )
                if requested:
                    wait_for_deleted(
                        cleanup_control_client.get_payment_manager,
                        "Payment Manager",
                        paymentManagerId=manager_id,
                    )

            if provider_name:
                requested = delete_or_skip(
                    cleanup_control_client.delete_payment_credential_provider,
                    "Credential provider and managed secret",
                    name=provider_name,
                )
                if requested:
                    wait_for_deleted(
                        cleanup_control_client.get_payment_credential_provider,
                        "Credential provider and managed secret",
                        name=provider_name,
                    )

        delete_payment_roles(cleanup_region)
        write_env({key: "" for key in cleanup_keys})
        for key in cleanup_keys:
            os.environ[key] = ""

        print("AWS cleanup complete.")
        print(
            "Next: revoke the Coinbase tutorial API key and optionally "
            "clear local Coinbase values below."
        )
        """
    ),
    _markdown(
        """
        ### Finish Coinbase and local cleanup

        Code in this repository cannot safely revoke your Coinbase credential.
        In Coinbase Developer Platform, open **API Keys → Secret API keys** and
        revoke the dedicated `agentcore-payments-tutorial` key.

        After revoking it, optionally run the next cell to blank the local
        Coinbase values and linked email in `.env`. It does not revoke or
        delete anything in Coinbase by itself.

        If you separately enabled AgentCore CloudWatch observability, review
        and remove its tutorial-specific log groups from CloudWatch. This
        example does not create those log groups directly.
        """
    ),
    _code(
        """
        import os

        from setup_utils import write_env

        EXPECTED_LOCAL_CLEAR_CONFIRMATION = "CLEAR LOCAL COINBASE VALUES"
        LOCAL_CLEAR_CONFIRMATION = ""
        if LOCAL_CLEAR_CONFIRMATION != EXPECTED_LOCAL_CLEAR_CONFIRMATION:
            raise RuntimeError(
                "Local cleanup is locked. Revoke the Coinbase tutorial API "
                "key first, then set LOCAL_CLEAR_CONFIRMATION to "
                f"{EXPECTED_LOCAL_CLEAR_CONFIRMATION!r} and rerun this cell."
            )

        local_coinbase_keys = [
            "COINBASE_API_KEY_ID",
            "COINBASE_API_KEY_SECRET",
            "COINBASE_WALLET_SECRET",
            "LINKED_EMAIL",
        ]
        write_env({key: "" for key in local_coinbase_keys})
        for key in local_coinbase_keys:
            os.environ[key] = ""
        print("Local Coinbase values cleared from .env.")
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
