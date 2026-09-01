"""Reference answers for the semantic-similarity and NER entity-correctness
metrics (testing-and-evaluation-plan.md's Q9 -- "an exact-text match against
a reference answer is too brittle... embed both the agent's answer and a
reference answer and require a minimum cosine similarity"). A reference
answer is a hand-written target MEANING, not a transcript of any specific
real run -- the same relationship RAGAS's context-precision/recall scoring
already has to golden_set.py's scenarios.

Decision Agent sample: reuses ragas_report.py's exact 5 scenario IDs
(consistent_low_00, partially_consistent_high_10, inconsistent_low_12,
change_of_mind_medium, guardrail_override_consistent_low) rather than a
new sample, for continuity and to avoid a second, separate real-API-cost
sample of the same golden set.

Image Parsing Agent sample: built once, pulled after a real, confirmed bug
(project-plan.md Q86 -- GPT-4.1 mini couldn't reliably reproduce a ~20K-
character base64 photo blob verbatim as a redact_photo tool-call
argument), and re-added here now that the real fix landed (Q86/Q87 --
analyze_claim_photo does the fetch/redact/analyze chain entirely server-
side, so the LLM never handles the raw photo bytes at all). Same 3 real
test images and DummyJSON order_refs as the original attempt -- the
scenario data itself was never the problem.
"""

from dataclasses import dataclass, field
from pathlib import Path

_IMAGES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "images"


@dataclass
class DecisionReferenceScenario:
    scenario_id: str  # must match a golden_set.py GoldenScenario.scenario_id
    reference_rationale: str
    expected_claim_categories: list = field(default_factory=list)
    expected_monetary_amounts: list = field(default_factory=list)


@dataclass
class ImageParsingReferenceScenario:
    reference_id: str
    order_ref: str
    claim_category: str
    claim_description: str
    photo_path: Path
    reference_reasoning: str
    expected_products: list = field(default_factory=list)
    expected_claim_categories: list = field(default_factory=list)


DECISION_REFERENCE_SCENARIOS = [
    DecisionReferenceScenario(
        scenario_id="consistent_low_00",
        reference_rationale=(
            "The photo evidence is consistent with the customer's Damaged in Transit claim and the "
            "fraud risk is low, so the claim is approved for a full refund to the original payment "
            "method, per the policy's Damaged in Transit and decision-matrix rules."
        ),
        expected_claim_categories=["Damaged in Transit"],
        expected_monetary_amounts=["$45.00"],
    ),
    DecisionReferenceScenario(
        scenario_id="partially_consistent_high_10",
        reference_rationale=(
            "Because the fraud risk is high, the claim is escalated to human review regardless of "
            "the partially consistent photo evidence, per the decision matrix's rule that any high "
            "fraud-risk band escalates."
        ),
        expected_claim_categories=["Not as Described"],
        expected_monetary_amounts=["$115.00"],
    ),
    DecisionReferenceScenario(
        scenario_id="inconsistent_low_12",
        reference_rationale=(
            "The photo evidence is inconsistent with the customer's claim, and even though fraud "
            "risk is low, the decision matrix escalates an inconsistent-and-low combination to human "
            "review rather than auto-denying or auto-approving it."
        ),
        expected_claim_categories=["Damaged in Transit"],
        expected_monetary_amounts=["$129.00"],
    ),
    DecisionReferenceScenario(
        scenario_id="change_of_mind_medium",
        reference_rationale=(
            "This is a Change of Mind claim within the 14-day window, so it is eligible for store "
            "credit only, not a refund to the original payment method; with consistent evidence and "
            "medium fraud risk, the claim is approved."
        ),
        expected_claim_categories=["Change of Mind"],
        expected_monetary_amounts=["$60.00"],
    ),
    DecisionReferenceScenario(
        scenario_id="guardrail_override_consistent_low",
        reference_rationale=(
            "Although the photo evidence is consistent and fraud risk is low, which would normally "
            "approve the claim, the refund amount of $350 exceeds the $200 guardrail threshold, so "
            "the claim must escalate to human review regardless of the matrix outcome."
        ),
        expected_claim_categories=["Damaged in Transit"],
        expected_monetary_amounts=["$350.00"],
    ),
]

IMAGE_PARSING_REFERENCE_SCENARIOS = [
    ImageParsingReferenceScenario(
        reference_id="damaged_item_transit",
        order_ref="1",
        claim_category="Damaged in Transit",
        claim_description="The item arrived with visible damage on the surface.",
        photo_path=_IMAGES_DIR / "damaged_item.jpg",
        reference_reasoning=(
            "The photo shows clear damage on the item, supporting the customer's claim that it "
            "arrived damaged in transit."
        ),
        expected_products=["Blue Frock"],
        expected_claim_categories=["Damaged in Transit"],
    ),
    ImageParsingReferenceScenario(
        reference_id="wrong_item_received",
        order_ref="1",
        claim_category="Wrong Item Received",
        claim_description="I ordered one item but received a completely different product.",
        photo_path=_IMAGES_DIR / "wrong_item.jpg",
        reference_reasoning=(
            "The photo shows a different product than the one that was ordered, confirming the "
            "customer received the wrong item."
        ),
        expected_products=["Blue Frock"],
        expected_claim_categories=["Wrong Item Received"],
    ),
    ImageParsingReferenceScenario(
        reference_id="damaged_shoes_transit",
        order_ref="4",
        claim_category="Damaged in Transit",
        claim_description="The shoes arrived damaged, with the sole visibly separated from the upper.",
        photo_path=_IMAGES_DIR / "damaged_shoes.jpg",
        reference_reasoning=(
            "The photo shows a shoe with visible damage at the sole, consistent with the customer's "
            "claim of damage in transit, and the product matches the ordered sneakers."
        ),
        expected_products=["Sports Sneakers Off White Red"],
        expected_claim_categories=["Damaged in Transit"],
    ),
]
