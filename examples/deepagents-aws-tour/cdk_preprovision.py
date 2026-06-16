"""Pre-provisioning CDK for the "Deep Agents on AWS" tour notebook.

Hand this single file to the AWS team. One `cdk deploy` stands up everything the
tour notebook (`deepagents_aws_tour.ipynb`) needs, so attendees
do not wait on infrastructure during the live session.

This is the trimmed footprint for the capability-tour notebook. It is deliberately
small:
this tour has NO Gateway/MCP federation path, so there are NO Lambdas, NO Cognito,
and NO AgentCore Gateway here. What the tour actually uses:

  - S3              -> the S3-backed durable filesystem route (Part 2) AND the
                       Bedrock KB data source. One bucket, two prefixes.
  - Bedrock KB      -> the agent's `query_product_kb` tool (Part 1), populated on
                       deploy so the KB is queryable immediately.
  - AgentCore       -> Code Interpreter sandbox backend (Part 3) and Browser
                       (optional). These are runtime API calls, not CFN resources,
                       so this template only grants the IAM permissions for them.
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
     - us.anthropic.claude-sonnet-4-6                (eval judge, Part 5)
     - amazon.titan-embed-text-v2:0                  (KB embeddings, Part 1)
2. Attach the managed policy this stack outputs (`AttendeePolicyArn`) to whatever
   identity attendees use (sandbox admin role, SSO permission set, or an IAM user).
   It grants bedrock:InvokeModel*, bedrock:Retrieve* on the KB, bedrock-agentcore:*
   (Code Interpreter + Browser), and S3 access to the bucket.
3. LangSmith is NOT AWS. Attendees create an account on the AWS-region instance
   (https://aws.smith.langchain.com), set LANGSMITH_API_KEY, and set
   LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com (the `api` host).
   Parts 5-7 (deploy via `langgraph deploy`, LangSmith eval templates, and UI
   review) are LangSmith-side and may need LangSmith Deployment access.

------------------------------------------------------------------------------
EFS note (why it's optional):
------------------------------------------------------------------------------
EFS only helps if attendees run in a container/host that can mount an NFS volume
(ECS/EKS/EC2). A Workshop Studio Jupyter or CloudShell environment generally
cannot, so the notebook falls back to a local temp dir to demonstrate the
FilesystemBackend pattern and narrates the EFS mount. Provision EFS only if your
attendee compute can mount it:
    cdk deploy -c include_efs=true
Leaving it off keeps the stack VPC-free and cheaper.

------------------------------------------------------------------------------
Stack outputs -> participant .env  (Workshop Studio can surface these):
------------------------------------------------------------------------------
  BedrockKbId        -> BEDROCK_KB_ID
  DataBucketName     -> AGENT_FILES_BUCKET   (S3 durable filesystem route)
  AttendeePolicyArn  -> attach to the attendee identity (see #2 above)
  EfsFileSystemId    -> EFS_FILE_SYSTEM_ID   (only if include_efs=true)
  EfsAccessPointId   -> EFS_ACCESS_POINT_ID  (only if include_efs=true)

------------------------------------------------------------------------------
To deploy (turn this single file into a CDK project):
------------------------------------------------------------------------------
  pip install "aws-cdk-lib>=2.150.0" "constructs>=10,<11" "cdklabs.generative-ai-cdk-constructs>=0.1.0"
  # cdk.json:  {"app": "uv run python cdk_preprovision.py"}
  # Keep the KB seed docs alongside this repo (they ship with it):
  #   data/   (product-doc markdown files used to populate the KB)
  cdk bootstrap aws://<account>/us-east-1
  cdk deploy                      # KB + S3 + IAM policy
  cdk deploy -c include_efs=true  # ...also an EFS filesystem + access point

------------------------------------------------------------------------------
Caveats to validate before relying on this:
------------------------------------------------------------------------------
  - The OpenSearch Serverless collection behind the KB is RETAINed on destroy by
    the cdklabs construct and bills continuously - decide on a teardown story for
    per-participant sandboxes.
  - `bedrock-agentcore:*` is broad for a sandbox; tighten to the specific Code
    Interpreter + Browser actions per AgentCore docs before any non-sandbox use.
"""
import pathlib

import aws_cdk as cdk
from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    Stack,
    aws_ec2 as ec2,
    aws_efs as efs,
    aws_iam as iam,
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
                    sid="AgentCoreCodeInterpreterAndBrowser",
                    # Code Interpreter (Part 3 sandbox backend) + Browser (optional tool).
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
