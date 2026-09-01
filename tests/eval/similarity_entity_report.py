"""Semantic similarity + NER entity-level correctness
(testing-and-evaluation-plan.md's Q9 -- "Semantic similarity" and "Named
entity recognition" use 1, "Entity-level correctness").

Purely informational, like ragas_report.py/cost_report.py -- Q18
explicitly defers setting similarity/entity-match thresholds until a real
score distribution exists to calibrate against, which is exactly what
running this produces. Not a pass/fail gate (unlike the PII-leak NER
check in test_ner_pii_leak.py, which Q27 makes non-negotiable regardless
of any threshold).

Real, not mocked: every scenario is a genuine Decision Agent or Image
Parsing Agent LLM call (tests/eval/scenario_runners.py), scored against a
hand-written reference answer (tests/eval/reference_answers.py) via a
genuine Azure OpenAI text-embedding-3-small call for cosine similarity,
and a genuine GPT-4.1 mini NER call (tests/eval/ner_utils.py) for
entity-level correctness.

Image Parsing coverage re-added here (project-plan.md Q87) after being
pulled in Q86 -- see reference_answers.py's module docstring for the full
history: a real, confirmed base64-truncation bug made every real Image
Parsing scenario fail, and the fix (Q86/Q87's analyze_claim_photo tool)
has since landed.

Run with: uv run python tests/eval/similarity_entity_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner_utils import cosine_similarity, embed, extract_entities  # noqa: E402
from reference_answers import DECISION_REFERENCE_SCENARIOS, IMAGE_PARSING_REFERENCE_SCENARIOS  # noqa: E402
from scenario_runners import run_decision_reasoning, run_image_parsing_reasoning  # noqa: E402


def _entity_recall(expected: list, extracted: list) -> float:
    """How many expected entities have a real match among the extracted
    ones -- case-insensitive substring containment either direction, since
    an LLM's real phrasing rarely echoes a reference string verbatim (e.g.
    extracted 'Sports Sneakers' should still count as matching expected
    'Sports Sneakers Off White Red')."""
    if not expected:
        return 1.0
    extracted_lower = [e.lower() for e in extracted]
    hits = 0
    for exp in expected:
        exp_lower = exp.lower()
        if any(exp_lower in e or e in exp_lower for e in extracted_lower):
            hits += 1
    return hits / len(expected)


def score_decision_scenarios() -> list:
    rows = []
    for ref in DECISION_REFERENCE_SCENARIOS:
        reasoning = run_decision_reasoning(ref.scenario_id)
        reasoning_vec, reference_vec = embed([reasoning, ref.reference_rationale])
        similarity = cosine_similarity(reasoning_vec, reference_vec)

        entities = extract_entities(reasoning)
        category_recall = _entity_recall(ref.expected_claim_categories, entities.claim_categories)
        amount_recall = _entity_recall(ref.expected_monetary_amounts, entities.monetary_amounts)

        rows.append({
            "scenario_id": ref.scenario_id, "agent": "Decision", "similarity": similarity,
            "category_recall": category_recall, "amount_recall": amount_recall,
        })
        print(
            f"[Decision] {ref.scenario_id:<40} similarity={similarity:.3f} "
            f"category_recall={category_recall:.2f} amount_recall={amount_recall:.2f}"
        )
    return rows


def score_image_parsing_scenarios() -> list:
    rows = []
    for ref in IMAGE_PARSING_REFERENCE_SCENARIOS:
        reasoning = run_image_parsing_reasoning(ref)
        reasoning_vec, reference_vec = embed([reasoning, ref.reference_reasoning])
        similarity = cosine_similarity(reasoning_vec, reference_vec)

        entities = extract_entities(reasoning)
        product_recall = _entity_recall(ref.expected_products, entities.products)
        category_recall = _entity_recall(ref.expected_claim_categories, entities.claim_categories)

        rows.append({
            "scenario_id": ref.reference_id, "agent": "Image Parsing", "similarity": similarity,
            "product_recall": product_recall, "category_recall": category_recall,
        })
        print(
            f"[Image Parsing] {ref.reference_id:<35} similarity={similarity:.3f} "
            f"product_recall={product_recall:.2f} category_recall={category_recall:.2f}"
        )
    return rows


def main() -> None:
    print("Scoring Decision Agent scenarios (semantic similarity + entity recall)...\n")
    decision_rows = score_decision_scenarios()
    print("\nScoring Image Parsing Agent scenarios (semantic similarity + entity recall)...\n")
    image_rows = score_image_parsing_scenarios()

    all_rows = decision_rows + image_rows
    all_similarities = [r["similarity"] for r in all_rows]
    print("\nReal score distribution (for threshold calibration, testing-and-evaluation-plan.md Q18):")
    print(
        f"  semantic_similarity: min={min(all_similarities):.3f} max={max(all_similarities):.3f} "
        f"avg={sum(all_similarities) / len(all_similarities):.3f}"
    )
    category_recalls = [r["category_recall"] for r in all_rows]
    print(f"  claim_category_entity_recall: avg={sum(category_recalls) / len(category_recalls):.3f}")
    product_recalls = [r["product_recall"] for r in image_rows]
    print(f"  product_entity_recall (Image Parsing only): avg={sum(product_recalls) / len(product_recalls):.3f}")


if __name__ == "__main__":
    main()
