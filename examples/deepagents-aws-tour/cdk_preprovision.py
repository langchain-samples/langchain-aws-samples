"""Pre-provisioning CDK for the "Deep Agents on AWS" tour notebook.

Hand this single file to the AWS team. One `cdk deploy` stands up everything the
tour notebook (`deepagents_aws_tour.ipynb`) needs, so attendees
do not wait on infrastructure during the live session.

This stack pre-provisions the AWS side of the 90-minute capability tour:

  - S3              -> the S3-backed durable filesystem route (Part 2) AND the
                       Bedrock KB data source. One bucket, two prefixes.
  - Bedrock KB      -> the agent's `query_product_kb` tool (Part 1), populated on
                       deploy so the KB is queryable immediately.
  - AgentCore       -> Code Interpreter sandbox backend, Browser, and Gateway/MCP.
                       Code Interpreter + Browser are runtime calls; Gateway is
                       created post-deploy by scripts/register_gateway.py.
  - Lambda/Cognito  -> order lookup, ticket history, refund target, and client-
                       credentials auth for the Gateway.
  - IAM user        -> optional hosted LangSmith Deployment runtime identity.
                       CDK creates the user and policy attachment, but not an
                       access key. scripts/create_deployment_user_key.py creates
                       one key and writes it to the local ignored .env file.
  - EFS (optional)  -> the "EFS as a pluggable backend" beat (Part 2). Off by
                       default; see the EFS note below.

Region is pinned to us-east-1 (AgentCore GA + the Claude models live there; the
`us.` model-id prefix is a cross-region inference profile and is load-bearing).

------------------------------------------------------------------------------
NOT covered by this template - AWS must also handle these out-of-band:
------------------------------------------------------------------------------
1. Bedrock model access (account-level enablement in us-east-1, not a CFN
   resource). Enable on-demand access for:
     - us.anthropic.claude-haiku-4-5-20251001-v1:0   (agent, all parts)
     - us.anthropic.claude-sonnet-4-6                (eval judge, Part 6)
     - amazon.titan-embed-text-v2:0                  (KB embeddings, Part 1)
2. Attach the managed policy this stack outputs (`AttendeePolicyArn`) to whatever
   identity attendees use (sandbox admin role, SSO permission set, or an IAM user).
   It grants bedrock:InvokeModel*, bedrock:Retrieve* on the KB, bedrock-agentcore:*
   (Code Interpreter, Browser, Gateway), and S3 access to the bucket.
3. LangSmith is NOT AWS. Attendees create an account on the AWS-region instance
   (https://aws.smith.langchain.com), set LANGSMITH_API_KEY, and set
   LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com (the `api` host).
   Parts 6-8 (LangSmith eval templates, optional UI-backed hosted deploy, and UI
   review) are LangSmith-side and may need LangSmith Deployment access.

------------------------------------------------------------------------------
EFS note (why it's optional):
------------------------------------------------------------------------------
EFS only helps if attendees run in a container/host that can mount an NFS volume
(ECS/EKS/EC2). A Workshop Studio Jupyter or CloudShell environment generally
cannot, so the notebook keeps EFS as a short pattern note and uses S3 as the
hands-on durable backend. Provision EFS only if your attendee compute can mount it:
    cdk deploy -c include_efs=true
Leaving it off keeps the stack VPC-free and cheaper.

------------------------------------------------------------------------------
Stack outputs -> participant .env  (Workshop Studio can surface these):
------------------------------------------------------------------------------
  BedrockKbId        -> BEDROCK_KB_ID
  DataBucketName     -> AGENT_FILES_BUCKET   (S3 durable filesystem route)
  PublicSupportDocKey -> PUBLIC_SUPPORT_DOC_KEY
  OrderLambdaArn / IssueLambdaArn / Cognito* / GatewayRoleArn -> used by
                       scripts/register_gateway.py
  HostedDeploymentUserName -> used by scripts/create_deployment_user_key.py
  HostedDeploymentUserArn -> IAM user for LangSmith Deployment runtime AWS creds
  AttendeePolicyArn  -> attach to the attendee identity (see #2 above)
  EfsFileSystemId    -> EFS_FILE_SYSTEM_ID   (only if include_efs=true)
  EfsAccessPointId   -> EFS_ACCESS_POINT_ID  (only if include_efs=true)

------------------------------------------------------------------------------
To deploy from this repo:
------------------------------------------------------------------------------
  uv sync --extra cdk --python 3.12
  cdk bootstrap aws://<account>/us-east-1
  cdk deploy                      # KB + S3 + IAM + Lambdas + Cognito
  uv run python scripts/register_gateway.py --write-env .env
  cdk deploy -c include_efs=true  # ...also an EFS filesystem + access point

Optional hosted AWS LangSmith UI deploy:
  uv run python scripts/create_deployment_user_key.py --write-env .env

Before stack teardown, if you created the optional hosted deployment key:
  uv run python scripts/create_deployment_user_key.py --delete-existing
  cdk destroy

------------------------------------------------------------------------------
Caveats to validate before relying on this:
------------------------------------------------------------------------------
  - The OpenSearch Serverless collection behind the KB is RETAINed on destroy by
    the cdklabs construct and bills continuously - decide on a teardown story for
    per-participant sandboxes.
  - `bedrock-agentcore:*` is broad for a sandbox; tighten to the specific Code
    Interpreter, Browser, and Gateway actions before any non-sandbox use.
  - The hosted deployment IAM user is for workshop convenience. Rotate or delete
    its access key after the workshop.
"""
import pathlib

import aws_cdk as cdk
from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_cognito as cognito,
    aws_ec2 as ec2,
    aws_efs as efs,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_s3 as s3,
    aws_s3_deployment as s3_deployment,
    custom_resources as cr,
)
from cdklabs.generative_ai_cdk_constructs import bedrock
from constructs import Construct

# KB seed docs ship with this standalone package. Referenced relative to this
# file so deploy works from any CWD.
_HERE = pathlib.Path(__file__).resolve().parent
KB_SEED_DATA_DIR = str(_HERE / "data")
PUBLIC_DOCS_DIR = str(_HERE / "public_docs")

# Models the attendee identity must be allowed to invoke (also enable access to
# these in the Bedrock console - see out-of-band #1). Listed here so the IAM
# policy and the model-access checklist can't drift apart.
AGENT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
JUDGE_MODEL = "us.anthropic.claude-sonnet-4-6"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"


class TourPreprovisionStack(Stack):
    """Everything the tour notebook needs, pre-provisioned in one stack."""

    def __init__(self, scope: Construct, construct_id: str, *, include_efs: bool, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # ======================================================================
        # S3: one bucket, used for both the KB data source and the agent's
        # S3-backed durable filesystem route (Part 2). DESTROY so per-attendee
        # sandboxes tear down cleanly.
        # ======================================================================
        bucket = s3.Bucket(
            self,
            "TourBucket",
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # ======================================================================
        # Bedrock Knowledge Base (Titan embeddings + OpenSearch Serverless),
        # backing the agent's query_product_kb tool (Part 1).
        # ======================================================================
        kb = bedrock.VectorKnowledgeBase(
            self,
            "ProductKb",
            embeddings_model=bedrock.BedrockFoundationModel.TITAN_EMBED_TEXT_V2_1024,
            instruction=(
                "Use this knowledge base to answer questions about product engineering "
                "issues, root causes, and documented fixes."
            ),
        )
        data_source = bedrock.S3DataSource(
            self,
            "ProductDocs",
            bucket=bucket,
            knowledge_base=kb,
            data_source_name="product-docs",
        )
        seed_data = s3_deployment.BucketDeployment(
            self,
            "SeedData",
            sources=[s3_deployment.Source.asset(KB_SEED_DATA_DIR)],
            destination_bucket=bucket,
            # Keep KB docs under a prefix so they don't collide with the agent's
            # S3 filesystem writes (which go under their own prefix at runtime).
            destination_key_prefix="kb-docs/",
        )
        public_doc_key = "public-docs/sh-hub-v2-troubleshooting.html"
        s3_deployment.BucketDeployment(
            self,
            "PublicSupportDocs",
            sources=[s3_deployment.Source.asset(PUBLIC_DOCS_DIR)],
            destination_bucket=bucket,
            destination_key_prefix="public-docs/",
        )

        # Bedrock does NOT ingest automatically - without this the KB stays empty
        # and query_product_kb returns "No matching product documentation found."
        start_ingestion = cr.AwsCustomResource(
            self,
            "StartIngestion",
            on_create=cr.AwsSdkCall(
                service="bedrock-agent",
                action="StartIngestionJob",
                parameters={
                    "knowledgeBaseId": kb.knowledge_base_id,
                    "dataSourceId": data_source.data_source_id,
                },
                physical_resource_id=cr.PhysicalResourceId.of("ProductKbInitialIngestion"),
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements([
                iam.PolicyStatement(
                    actions=["bedrock:StartIngestionJob"],
                    resources=[kb.knowledge_base_arn],
                )
            ]),
        )
        start_ingestion.node.add_dependency(seed_data)
        start_ingestion.node.add_dependency(data_source)

        # ======================================================================
        # Lambda + Cognito: Gateway targets and client-credentials auth. The
        # Gateway itself is created post-deploy so the script can stay idempotent
        # and fetch the Cognito client secret without exposing it as a CFN output.
        # ======================================================================
        order_fn = _lambda.Function(
            self,
            "OrderManagementFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            code=_lambda.Code.from_asset(str(_HERE / "lambdas" / "order_management")),
            handler="handler.lambda_handler",
            timeout=Duration.seconds(30),
            memory_size=256,
        )
        issue_fn = _lambda.Function(
            self,
            "IssueManagementFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            code=_lambda.Code.from_asset(str(_HERE / "lambdas" / "issue_management")),
            handler="handler.lambda_handler",
            timeout=Duration.seconds(30),
            memory_size=256,
        )

        pool = cognito.UserPool(
            self,
            "GatewayAuthPool",
            user_pool_name="deepagents-tour-gateway-pool",
            self_sign_up_enabled=False,
            removal_policy=RemovalPolicy.DESTROY,
        )
        domain = pool.add_domain(
            "Domain",
            cognito_domain=cognito.CognitoDomainOptions(domain_prefix=f"deepagents-tour-{self.account}"),
        )
        invoke_scope = cognito.ResourceServerScope(
            scope_name="invoke",
            scope_description="Invoke MCP Gateway tools",
        )
        resource_server = pool.add_resource_server(
            "GatewayResourceServer",
            identifier="mcp-gateway",
            scopes=[invoke_scope],
        )
        client = pool.add_client(
            "GatewayClient",
            generate_secret=True,
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(client_credentials=True),
                scopes=[cognito.OAuthScope.resource_server(resource_server, invoke_scope)],
            ),
        )

        gateway_role = iam.Role(
            self,
            "GatewayRole",
            assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
            description="Role assumed by AgentCore Gateway to invoke Lambda targets",
        )
        order_fn.grant_invoke(gateway_role)
        issue_fn.grant_invoke(gateway_role)

        # ======================================================================
        # Attendee permissions. AgentCore Code Interpreter/Browser are runtime
        # API calls (no CFN resource), so we only grant the IAM. Emitted as a
        # managed policy the AWS team attaches to the attendee identity (see
        # out-of-band #2) rather than a role, since sandbox identities vary.
        # ======================================================================
        attendee_policy = iam.ManagedPolicy(
            self,
            "AttendeePolicy",
            description="Permissions the tour notebook needs at runtime (attach to attendee identity)",
            statements=[
                iam.PolicyStatement(
                    sid="BedrockInvoke",
                    actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                    # Broad on purpose for a sandbox; tighten to the specific model +
                    # inference-profile ARNs (AGENT_MODEL/JUDGE_MODEL/EMBED_MODEL) for prod.
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    sid="BedrockKbRetrieve",
                    actions=["bedrock:Retrieve", "bedrock:RetrieveAndGenerate"],
                    resources=[kb.knowledge_base_arn],
                ),
                iam.PolicyStatement(
                    sid="AgentCoreRuntimeAndGateway",
                    # Code Interpreter, Browser, Gateway control, and MCP runtime calls.
                    actions=["bedrock-agentcore:*"],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    sid="S3DurableFilesystem",
                    actions=[
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject",
                        "s3:ListBucket",
                    ],
                    resources=[bucket.bucket_arn, f"{bucket.bucket_arn}/*"],
                ),
            ],
        )
        hosted_deployment_user = iam.User(
            self,
            "HostedDeploymentUser",
            user_name=f"deepagents-tour-deployment-{self.account}",
        )
        attendee_policy.attach_to_user(hosted_deployment_user)

        # ======================================================================
        # EFS (optional) - the "EFS as a pluggable backend" beat. Needs a VPC +
        # mount targets; only useful if attendee compute can mount NFS. Off by
        # default (see the EFS note in the module docstring).
        # ======================================================================
        if include_efs:
            vpc = ec2.Vpc(
                self,
                "TourVpc",
                max_azs=2,
                nat_gateways=0,  # EFS mount targets sit in isolated subnets; no NAT needed
                subnet_configuration=[
                    ec2.SubnetConfiguration(
                        name="isolated", subnet_type=ec2.SubnetType.PRIVATE_ISOLATED, cidr_mask=24
                    )
                ],
            )
            file_system = efs.FileSystem(
                self,
                "TourEfs",
                vpc=vpc,
                removal_policy=RemovalPolicy.DESTROY,
            )
            # Allow NFS from within the VPC so the attendee compute can mount it.
            file_system.connections.allow_default_port_from(
                ec2.Peer.ipv4(vpc.vpc_cidr_block), "NFS from within the VPC"
            )
            access_point = file_system.add_access_point(
                "TourAccessPoint",
                path="/memories",
                create_acl=efs.Acl(owner_uid="1000", owner_gid="1000", permissions="0755"),
                posix_user=efs.PosixUser(uid="1000", gid="1000"),
            )
            CfnOutput(self, "EfsFileSystemId", value=file_system.file_system_id, description="-> EFS_FILE_SYSTEM_ID")
            CfnOutput(self, "EfsAccessPointId", value=access_point.access_point_id, description="-> EFS_ACCESS_POINT_ID")
            CfnOutput(self, "TourVpcId", value=vpc.vpc_id)

        # ======================================================================
        # Outputs -> participant .env (see the mapping in the module docstring)
        # ======================================================================
        CfnOutput(self, "BedrockKbId", value=kb.knowledge_base_id, description="-> BEDROCK_KB_ID")
        CfnOutput(self, "DataBucketName", value=bucket.bucket_name, description="-> AGENT_FILES_BUCKET")
        CfnOutput(self, "PublicSupportDocKey", value=public_doc_key, description="-> PUBLIC_SUPPORT_DOC_KEY")
        CfnOutput(self, "OrderLambdaArn", value=order_fn.function_arn)
        CfnOutput(self, "IssueLambdaArn", value=issue_fn.function_arn)
        CfnOutput(self, "CognitoUserPoolId", value=pool.user_pool_id)
        CfnOutput(
            self,
            "CognitoTokenUrl",
            value=f"https://{domain.domain_name}.auth.{self.region}.amazoncognito.com/oauth2/token",
        )
        CfnOutput(self, "CognitoClientId", value=client.user_pool_client_id)
        CfnOutput(self, "GatewayRoleArn", value=gateway_role.role_arn)
        CfnOutput(self, "HostedDeploymentUserName", value=hosted_deployment_user.user_name)
        CfnOutput(self, "HostedDeploymentUserArn", value=hosted_deployment_user.user_arn)
        CfnOutput(
            self,
            "AttendeePolicyArn",
            value=attendee_policy.managed_policy_arn,
            description="Attach to the attendee identity (out-of-band #2)",
        )


app = cdk.App()
# `cdk deploy -c include_efs=true` to also provision EFS (+ a VPC).
_include_efs = str(app.node.try_get_context("include_efs")).lower() in ("1", "true", "yes")
TourPreprovisionStack(
    app,
    "TourPreprovisionStack",
    include_efs=_include_efs,
    # Pinned to us-east-1; account comes from the deploying environment.
    env=cdk.Environment(region="us-east-1"),
)
app.synth()
