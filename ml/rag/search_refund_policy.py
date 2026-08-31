"""The real search_refund_policy(query) implementation (project-plan.md
Q50/Q54/Q56) -- what the Decision MCP server's tool will wrap once that server
exists. k-NN search against the real OpenSearch Serverless index, top-5,
returning each chunk's text + policy_version."""

import os

from dotenv import load_dotenv

from embed_and_index import INDEX_NAME, embed_texts, get_azure_client
from index_chunks import get_opensearch_client

load_dotenv()

TOP_K = 5  # project-plan.md Q56

# Q56's "no confident match -> escalate" fallback, calibrated against real
# data (ml/rag/evaluate_retrieval.py's 16-query set vs. 4 clearly off-topic
# queries): on-topic top-1 scores ranged 0.6249-0.8570; off-topic top-1
# scores ranged 0.5568-0.6135. 0.62 sits in that real gap, not guessed.
MIN_CONFIDENT_SCORE = 0.62


def search_refund_policy(query: str, top_k: int = TOP_K) -> list[dict]:
    azure_client = get_azure_client()
    query_vector = embed_texts(azure_client, [query])[0]

    os_client = get_opensearch_client()
    response = os_client.search(
        index=INDEX_NAME,
        body={
            "size": top_k,
            "query": {"knn": {"embedding": {"vector": query_vector, "k": top_k}}},
        },
    )
    return [
        {
            "chunk_id": hit["_source"]["chunk_id"],
            "title": hit["_source"]["title"],
            "text": hit["_source"]["text"],
            "policy_version": hit["_source"]["policy_version"],
            "score": hit["_score"],
        }
        for hit in response["hits"]["hits"]
    ]


def search_refund_policy_or_escalate(query: str, top_k: int = TOP_K) -> dict:
    """What the Decision Agent actually calls: Q56's no-confident-match fallback.
    Escalates rather than reasoning from a weak/irrelevant retrieval."""
    results = search_refund_policy(query, top_k=top_k)
    if not results or results[0]["score"] < MIN_CONFIDENT_SCORE:
        return {
            "confident": False,
            "action": "escalate",
            "reason": "no confidently relevant policy chunk found for this claim",
            "top_score": results[0]["score"] if results else None,
        }
    return {"confident": True, "action": "use_chunks", "chunks": results}


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What happens if my refund is over $200?"
    print(f"Query: {query!r}\n")
    for i, result in enumerate(search_refund_policy(query), 1):
        print(f"{i}. {result['chunk_id']} (score={result['score']:.4f}, {result['policy_version']})")
        print(f"   {result['text'][:100]}...")

    print("\nsearch_refund_policy_or_escalate:")
    print(search_refund_policy_or_escalate(query))
