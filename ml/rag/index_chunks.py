"""Creates the k-NN index on the real OpenSearch Serverless collection and
bulk-indexes the embedded policy chunks (project-plan.md Q50/Q54/Q56)."""

import os
import time

import boto3
from dotenv import load_dotenv
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

from embed_and_index import INDEX_NAME, main as embed_chunks
from provision_opensearch import COLLECTION_NAME, client as aoss_client

load_dotenv()
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
EMBEDDING_DIM = 1536  # text-embedding-3-small


def get_opensearch_client() -> OpenSearch:
    details = aoss_client.batch_get_collection(names=[COLLECTION_NAME])["collectionDetails"]
    if not details:
        raise RuntimeError(
            f"OpenSearch Serverless collection '{COLLECTION_NAME}' doesn't exist -- "
            f"run ml/rag/provision_opensearch.py (and ml/rag/index_chunks.py) first."
        )
    endpoint = details[0]["collectionEndpoint"]
    host = endpoint.replace("https://", "")
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, AWS_REGION, "aoss")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
        timeout=30,
    )


def ensure_index(client: OpenSearch):
    if client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists")
        return
    client.indices.create(
        index=INDEX_NAME,
        body={
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIM,
                        "method": {"name": "hnsw", "engine": "nmslib", "space_type": "cosinesimil"},
                    },
                    "chunk_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "text": {"type": "text"},
                    "policy_version": {"type": "keyword"},
                    "source_document": {"type": "keyword"},
                }
            },
        },
    )
    print(f"Created index '{INDEX_NAME}' (knn_vector dim={EMBEDDING_DIM}, cosine)")


def index_chunks(client: OpenSearch, chunks: list[dict]):
    # OpenSearch Serverless (vector-search collections) rejects a caller-supplied
    # document _id on index -- rely on the chunk_id *field* for identity instead.
    for chunk in chunks:
        body = {
            "chunk_id": chunk["chunk_id"],
            "title": chunk["title"],
            "text": chunk["text"],
            "policy_version": chunk["policy_version"],
            "embedding": chunk["embedding"],
        }
        # project-plan.md Q96: chunk_document.py's admin-upload chunks carry
        # a source_document field that chunk_policy.py's refund-policy
        # chunks don't have -- included only when present so the base
        # policy chunks' documents are unchanged.
        if "source_document" in chunk:
            body["source_document"] = chunk["source_document"]
        client.index(index=INDEX_NAME, body=body)
    print(f"Indexed {len(chunks)} chunks into '{INDEX_NAME}'")
    # OpenSearch Serverless doesn't support a synchronous refresh=True request
    # (rejected above) -- it refreshes on its own interval instead. Measured
    # empirically: not yet visible at 10s, visible by 20s.
    time.sleep(20)


if __name__ == "__main__":
    chunks_with_embeddings = embed_chunks()
    os_client = get_opensearch_client()
    ensure_index(os_client)
    index_chunks(os_client, chunks_with_embeddings)

    count = os_client.count(index=INDEX_NAME)["count"]
    print(f"\nVerified: index '{INDEX_NAME}' now contains {count} documents")
