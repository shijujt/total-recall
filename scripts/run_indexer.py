from pathlib import Path

from ir.indexer import AwsSvcIndexer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
ASSETS_PATH = PROJECT_ROOT / "assets"
EVAL_OUTPUT = PROJECT_ROOT / "eval_queries_ag.jsonl"

SVC_NAMES = [
    "aws-lambda-developer-guide",
    "amazon-s3-developer-guide",
    "amazon-s3-getting-started-guide",
    "amazon-s3-user-guide",
    "amazon-dynamodb-developer-guide",
    "aws-dynamodb-encryption-docs",
    "amazon-api-gateway-developer-guide",
    "amazon-ecs-developer-guide",
    "amazon-elasticsearch-service-developer-guide",
    "amazon-eventbridge-user-guide",
    "amazon-kendra-developer-guide",
    "amazon-sagemaker-developer-guide",
    "amazon-sns-developer-guide",
    "amazon-sqs-developer-guide",
    "aws-cli-user-guide",
    "aws-glue-developer-guide",
    "aws-secrets-manager-docs",
    "iam-user-guide",
    "amazon-opensearch-service-developer-guide",
    "amazon-elasticache-docs",
    "aws-cloudformation-user-guide",
    "aws-data-pipeline-developer-guide",
    "aws-step-functions-developer-guide",
]

# Set collection_name to "aws_docs" to index, or None to generate eval queries
COLLECTION_NAME = "aws_docs"

indexer = AwsSvcIndexer(
    base_docs_path=str(ASSETS_PATH),
    collection_name=COLLECTION_NAME,
    chroma_path=str(CHROMA_PATH),
)
indexer.index_services(SVC_NAMES, eval_output_path=str(EVAL_OUTPUT))
