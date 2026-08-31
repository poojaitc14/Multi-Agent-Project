"""Embed the 8 policy chunks (Azure OpenAI text-embedding-3-small, per
project-plan.md Q50) and index them into OpenSearch Serverless (Q54).

Collection/index naming: dev-prefixed, matching the DynamoDB/S3 convention
project-plan.md Q51 already established for local dev against real AWS.
"""

import os

from dotenv import load_dotenv
from openai import AzureOpenAI

from chunk_policy import POLICY_PATH, chunk_policy

load_dotenv()

INDEX_NAME = os.environ.get("OPENSEARCH_REFUND_POLICY_INDEX", "refund-policy-dev")


def get_azure_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def embed_texts(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
    response = client.embeddings.create(model=deployment, input=texts)
    return [item.embedding for item in response.data]


def main():
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    chunks = chunk_policy(policy_text)
    print(f"Chunked policy {chunks[0]['policy_version']} into {len(chunks)} chunks")

    client = get_azure_client()
    embeddings = embed_texts(client, [c["text"] for c in chunks])
    print(f"Embedded {len(embeddings)} chunks, dimension={len(embeddings[0])}")

    for chunk, vector in zip(chunks, embeddings):
        chunk["embedding"] = vector

    return chunks


if __name__ == "__main__":
    result = main()
    print(f"\nReady to index {len(result)} chunks into OpenSearch Serverless index '{INDEX_NAME}'")
