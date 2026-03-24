import numpy as np
from sentence_transformers import SentenceTransformer

SERVICE_DESCRIPTIONS = {
    "s3": "s3 bucket buckets object objects key prefix policy storage lifecycle versioning replication",  # noqa: E501
    "lambda": "lambda function functions runtime handler timeout execution environment variables layers",  # noqa: E501
    "dynamodb": "dynamodb table tables partition key sort key nosql indexes throughput gsi lsi",
    "sns": "sns topic topics publish subscribe notification message",
    "sqs": "sqs queue queues message visibility timeout polling",
    "step-functions": "step functions state machine workflow execution",
    "api-gateway": "api gateway rest api http api endpoint stage deployment",
    "glue": "glue data catalog crawler etl spark job database table",
    "iam": "iam role roles policy policies permission permissions user users execution role",
    "cloudformation": "cloudformation template stack stacks resource",
    "cli": "aws cli command commands profile configuration",
}


class ServicePredictor:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.services = list(SERVICE_DESCRIPTIONS.keys())
        texts = list(SERVICE_DESCRIPTIONS.values())
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def predict(self, query, top_k=2):
        q_emb = self.model.encode(query, normalize_embeddings=True)
        scores = np.dot(self.embeddings, q_emb)

        ranked = sorted(
            zip(self.services, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_k]
