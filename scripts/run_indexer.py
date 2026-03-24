import ir.config as cfg
from ir.indexer import AwsSvcIndexer

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

# Set collection_name to cfg.COLLECTION_NAME to index, or None to generate eval queries
indexer = AwsSvcIndexer(
    base_docs_path=str(cfg.ASSETS_PATH),
    collection_name=cfg.COLLECTION_NAME,
    chroma_path=str(cfg.CHROMA_PATH),
)
indexer.index_services(SVC_NAMES, eval_output_path=str(cfg.EVAL_OUTPUT_FILE))
