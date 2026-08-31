"""Evaluates search_refund_policy retrieval quality against the real
OpenSearch Serverless index -- real embeddings, real k-NN search, no
mocking. 16 hand-written queries (2 per chunk, all 8 chunks covered),
phrased the way a customer or the Decision Agent's own reasoning might phrase
them, not just the chunk's own header text repeated back.

Metrics: hit@5 (is the expected chunk anywhere in the top 5 -- this is
what actually matters, since Q56 set top_k=5), MRR (how highly it ranks
within those 5), and hit@1 (stricter: is it literally the top result).
"""

from search_refund_policy import search_refund_policy

EVAL_SET = [
    ("How many days do I have to return something?", "return_window"),
    ("Is the return window different for change of mind vs a defective item?", "return_window"),
    ("My package arrived with a broken screen, what category is that?", "damaged_in_transit"),
    ("The box was crushed and the item inside is smashed", "damaged_in_transit"),
    ("I ordered a blue shirt but got a red one instead", "wrong_item_received"),
    ("They sent me the wrong size shoes", "wrong_item_received"),
    ("The listing said leather but this feels like plastic", "not_as_described"),
    ("Product works but doesn't match the photos on the website", "not_as_described"),
    ("The item doesn't turn on at all, straight out of the box", "defective_doa"),
    ("It's brand new but completely non-functional", "defective_doa"),
    ("I just don't want this anymore, nothing is wrong with it", "change_of_mind"),
    ("Can I get store credit if I simply changed my mind?", "change_of_mind"),
    ("How does the system decide between approve, deny, and escalate?", "decision_matrix"),
    ("What's the outcome for a consistent photo with high fraud risk?", "decision_matrix"),
    ("Is there a dollar amount that always requires a human to review it?", "guardrail"),
    ("What happens for refunds above two hundred dollars?", "guardrail"),
]


def evaluate():
    hits_at_5 = 0
    hits_at_1 = 0
    reciprocal_ranks = []
    rows = []

    for query, expected_chunk in EVAL_SET:
        results = search_refund_policy(query)
        retrieved_ids = [r["chunk_id"] for r in results]

        if expected_chunk in retrieved_ids:
            hits_at_5 += 1
            rank = retrieved_ids.index(expected_chunk) + 1
            reciprocal_ranks.append(1 / rank)
            if rank == 1:
                hits_at_1 += 1
        else:
            rank = None
            reciprocal_ranks.append(0.0)

        rows.append((query, expected_chunk, retrieved_ids[0], rank))

    n = len(EVAL_SET)
    print(f"{'query':<58} {'expected':<22} {'top result':<22} {'rank'}")
    print("-" * 115)
    for query, expected, top_result, rank in rows:
        marker = "OK" if rank == 1 else ("~" if rank else "MISS")
        print(f"{query[:56]:<58} {expected:<22} {top_result:<22} {rank or '-':<6} {marker}")

    print()
    print(f"hit@1 (expected chunk is the top result):  {hits_at_1}/{n} = {hits_at_1/n:.1%}")
    print(f"hit@5 (expected chunk is in the top 5):     {hits_at_5}/{n} = {hits_at_5/n:.1%}")
    print(f"MRR (mean reciprocal rank):                 {sum(reciprocal_ranks)/n:.3f}")


if __name__ == "__main__":
    evaluate()
