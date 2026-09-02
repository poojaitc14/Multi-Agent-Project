"""The real search_refund_policy(query) implementation (project-plan.md
Q50/Q54/Q56) -- what the Decision MCP server's tool will wrap once that server
exists. Retrieve-then-rerank pipeline against the real OpenSearch Serverless
index: a k-NN (semantic) query and a BM25 match (lexical) query on the same
"text" field, merged by Reciprocal Rank Fusion into a candidate pool, then
reranked by a real cross-encoder (joint query+passage scoring, more accurate
than bi-encoder/BM25 alone) down to the final top-10. Pure semantic search
can miss exact policy terms (dollar thresholds, category names) that don't
embed distinctively; BM25 catches those, and the cross-encoder resolves
which of the combined candidates are truly most relevant. Hybrid retrieval
implemented as two separate queries merged in Python rather than
OpenSearch's native hybrid-query search-pipeline feature, since that
feature's availability on Serverless collections isn't confirmed -- this
approach works regardless."""

import os
from functools import lru_cache

from dotenv import load_dotenv

from embed_and_index import INDEX_NAME, embed_texts, get_azure_client
from index_chunks import get_opensearch_client

load_dotenv()

TOP_K = 10  # project-plan.md Q56 -- raised from 5 to give recall more room
RETRIEVAL_K = 15  # candidate pool per branch (k-NN, BM25) feeding the reranker

# Q56's "no confident match -> escalate" fallback, calibrated against real
# data (ml/rag/evaluate_retrieval.py's 16-query set vs. 4 clearly off-topic
# queries): on-topic top-1 scores ranged 0.6249-0.8570; off-topic top-1
# scores ranged 0.5568-0.6135. 0.62 sits in that real gap, not guessed.
# Calibrated against pure k-NN cosine scores -- so the confidence gate below
# always checks the k-NN top-1 score directly, never the hybrid-merged or
# reranked result order, even though those decide what's actually *returned*.
MIN_CONFIDENT_SCORE = 0.62

RRF_K = 60  # standard Reciprocal Rank Fusion constant
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Loaded once per process -- the model itself is ~80MB, small next to
    the fraud model / EasyOCR detectors already warmed up at mcp-server
    startup."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(CROSS_ENCODER_MODEL)


def _reciprocal_rank_fusion(knn_hits: list[dict], bm25_hits: list[dict]) -> list[dict]:
    """Merge two OpenSearch hit lists by Reciprocal Rank Fusion: each chunk's
    combined score is the sum of 1/(RRF_K + rank + 1) across whichever list(s)
    it appears in, rewarding chunks either method ranks highly. Returns the
    full merged candidate pool, unranked-by-truncation -- the cross-encoder
    does the final ranking/truncation, this just assembles candidates. A
    chunk found only via BM25 keeps its real BM25 relevance score in "score"
    (not a cosine value) -- "score" here is per-chunk provenance, not the
    ranking signal."""
    rrf_scores: dict[str, float] = {}
    chunk_hits: dict[str, dict] = {}
    for hits in (knn_hits, bm25_hits):
        for rank, hit in enumerate(hits):
            chunk_id = hit["_source"]["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunk_hits.setdefault(chunk_id, hit)

    ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {
            "chunk_id": chunk_id,
            "title": chunk_hits[chunk_id]["_source"]["title"],
            "text": chunk_hits[chunk_id]["_source"]["text"],
            "policy_version": chunk_hits[chunk_id]["_source"]["policy_version"],
            "score": chunk_hits[chunk_id]["_score"],
            "rrf_score": round(rrf_score, 6),
        }
        for chunk_id, rrf_score in ranked
    ]


def _cross_encoder_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Reranks the RRF candidate pool with a real cross-encoder -- it scores
    (query, chunk_text) jointly rather than comparing independently-computed
    vectors, which is why it's more accurate than the bi-encoder/BM25 signals
    that built the candidate pool in the first place."""
    if not candidates:
        return candidates
    model = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in candidates]
    ce_scores = model.predict(pairs)
    for c, s in zip(candidates, ce_scores):
        c["cross_encoder_score"] = float(s)
    return sorted(candidates, key=lambda c: c["cross_encoder_score"], reverse=True)[:top_k]


def _hybrid_search(query: str, top_k: int) -> tuple[list[dict], float | None]:
    """Runs the k-NN and BM25 queries once (each over RETRIEVAL_K candidates),
    RRF-merges them, cross-encoder-reranks down to top_k, and returns the
    pure k-NN top-1 score alongside (for the Q56 confidence gate)."""
    azure_client = get_azure_client()
    query_vector = embed_texts(azure_client, [query])[0]
    os_client = get_opensearch_client()

    knn_hits = os_client.search(
        index=INDEX_NAME,
        body={
            "size": RETRIEVAL_K,
            "query": {"knn": {"embedding": {"vector": query_vector, "k": RETRIEVAL_K}}},
        },
    )["hits"]["hits"]
    bm25_hits = os_client.search(
        index=INDEX_NAME,
        body={
            "size": RETRIEVAL_K,
            "query": {"match": {"text": query}},
        },
    )["hits"]["hits"]

    knn_top1_score = knn_hits[0]["_score"] if knn_hits else None
    candidates = _reciprocal_rank_fusion(knn_hits, bm25_hits)
    reranked = _cross_encoder_rerank(query, candidates, top_k)
    return reranked, knn_top1_score


def search_refund_policy(query: str, top_k: int = TOP_K) -> list[dict]:
    results, _ = _hybrid_search(query, top_k)
    return results


def search_refund_policy_or_escalate(query: str, top_k: int = TOP_K) -> dict:
    """What the Decision Agent actually calls: Q56's no-confident-match fallback.
    Escalates rather than reasoning from a weak/irrelevant retrieval."""
    results, knn_top1_score = _hybrid_search(query, top_k)
    if not results or knn_top1_score is None or knn_top1_score < MIN_CONFIDENT_SCORE:
        return {
            "confident": False,
            "action": "escalate",
            "reason": "no confidently relevant policy chunk found for this claim",
            "top_score": knn_top1_score,
        }
    return {"confident": True, "action": "use_chunks", "chunks": results}


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What happens if my refund is over $200?"
    print(f"Query: {query!r}\n")
    for i, result in enumerate(search_refund_policy(query), 1):
        ce = result.get("cross_encoder_score")
        print(f"{i}. {result['chunk_id']} (score={result['score']:.4f}, cross_encoder={ce:.4f}, {result['policy_version']})")
        print(f"   {result['text'][:100]}...")

    print("\nsearch_refund_policy_or_escalate:")
    print(search_refund_policy_or_escalate(query))
