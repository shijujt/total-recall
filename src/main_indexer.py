svc_names = [
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
             "aws-step-functions-developer-guide"
            ]
svc_path =  "./assets"
c_nm = "aws_docs"
c_nm = None
indexer = AwsSvcIndexer(base_docs_path=svc_path, collection_name=c_nm)
indexer.index_services(svc_names)

