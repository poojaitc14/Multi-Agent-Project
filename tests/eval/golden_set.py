"""The golden set (testing-and-evaluation-plan.md's "Eval harness design"):
25-50 hand-crafted claims, each with a known expected verdict derived
directly from the real decision matrix in mcp-servers/orchestrator_server.py
(docs/refund_policy.md's "Decision matrix" + the >$200-always-escalate
guardrail).

Scoped to the Decision Agent specifically (image_verdict/fraud_risk_band
are given, not produced by a real upstream Image Parsing/Fraud Scoring
run) -- the same shape already used to get this project's first genuine,
non-fabricated Decision Agent verdict (project-plan.md Q75/Q81). Running
the full 4-agent pipeline per scenario would make "expected verdict"
non-deterministic for reasons outside this harness's control: Fraud
Scoring's synthetic risk profile is bootstrap-sampled per new customer_ref
(mcp-servers/orchestrator_server.py's get_account_info docstring), so the
same claim inputs can genuinely land on a different fraud_risk_band run to
run. Testing Image Parsing's and Fraud Scoring's own output quality is a
separate concern from testing whether the Decision Agent correctly reports
what a KNOWN (image_verdict, fraud_risk_band, refund_amount_usd) triple
resolves to.

_DECISION_MATRIX and _HIGH_VALUE_ESCALATION_THRESHOLD_USD below are a
deliberate, literal mirror of the real constants in
mcp-servers/orchestrator_server.py, not a reimplementation -- if they ever
drift apart, test_golden_set.py's own test_mirrored_decision_matrix_matches_
the_real_tool catches it directly, rather than this file silently trusting
a stale copy.
"""

from dataclasses import dataclass, field

_DECISION_MATRIX = {
    ("consistent", "low"): "approve",
    ("consistent", "medium"): "approve",
    ("consistent", "high"): "escalate",
    ("partially_consistent", "low"): "approve",
    ("partially_consistent", "medium"): "escalate",
    ("partially_consistent", "high"): "escalate",
    ("inconsistent", "low"): "escalate",
    ("inconsistent", "medium"): "deny",
    ("inconsistent", "high"): "deny",
}
_HIGH_VALUE_ESCALATION_THRESHOLD_USD = 200.0


def expected_decision(image_verdict: str, fraud_risk_band: str, refund_amount_usd: float) -> str:
    if refund_amount_usd > _HIGH_VALUE_ESCALATION_THRESHOLD_USD:
        return "escalate"
    return _DECISION_MATRIX[(image_verdict, fraud_risk_band)]


# Tool routing accuracy / tool call sequence (testing-and-evaluation-plan.md
# Q9's "Agent evaluation metrics") -- shared between test_golden_set.py's
# per-scenario gating assertions and ragas_report.py's measured-score
# reporting, so both read from one real TaskOutput.messages trace per
# scenario rather than each defining its own copy.
EXPECTED_DECISION_AGENT_TOOLS = {"search_refund_policy", "apply_decision_matrix"}


def tool_call_sequence(messages: list) -> list:
    return [m["name"] for m in messages if m.get("role") == "tool" and m.get("name")]


@dataclass
class GoldenScenario:
    scenario_id: str
    claim_category: str
    claim_description: str
    refund_amount_usd: float
    image_verdict: str
    fraud_risk_band: str
    fraud_key_signals: list = field(default_factory=list)

    @property
    def claim_ref(self) -> str:
        return f"clm_golden_{self.scenario_id}"

    @property
    def expected_decision(self) -> str:
        return expected_decision(self.image_verdict, self.fraud_risk_band, self.refund_amount_usd)


_PHOTO_REQUIRED_CATEGORIES = ("Damaged in Transit", "Wrong Item Received", "Not as Described", "Defective/DOA")
_DESCRIPTIONS = {
    "Damaged in Transit": "The box arrived crushed and the item inside is visibly cracked.",
    "Wrong Item Received": "I ordered a blue jacket but received a red scarf instead.",
    "Not as Described": "The listing said genuine leather, but this is clearly synthetic material.",
    "Defective/DOA": "The device won't power on at all, straight out of the box.",
    "Change of Mind": "I just don't want it anymore, it's still unopened.",
}
_IMAGE_VERDICTS = ("consistent", "partially_consistent", "inconsistent")
_FRAUD_BANDS = ("low", "medium", "high")


def build_golden_set() -> list[GoldenScenario]:
    """9 real decision-matrix cells x 4 photo-required categories (rotated,
    not every cell x every category, to stay in the spec's 25-50 range
    while still covering every category and every cell at least twice) +
    Change of Mind's 3 fraud bands (image_verdict is always 'consistent'
    for it, Q65 -- no photo path exists) + a handful of explicit >$200
    guardrail-override cases layered on top of a few of those same cells.
    36 scenarios total."""
    scenarios: list[GoldenScenario] = []

    cell_index = 0
    for image_verdict in _IMAGE_VERDICTS:
        for fraud_risk_band in _FRAUD_BANDS:
            for _ in range(2):  # each cell appears twice, with a different category
                category = _PHOTO_REQUIRED_CATEGORIES[cell_index % len(_PHOTO_REQUIRED_CATEGORIES)]
                cell_index += 1
                scenarios.append(
                    GoldenScenario(
                        scenario_id=f"{image_verdict}_{fraud_risk_band}_{len(scenarios):02d}",
                        claim_category=category,
                        claim_description=_DESCRIPTIONS[category],
                        refund_amount_usd=45.0 + (len(scenarios) * 7 % 140),  # varied, always under $200
                        image_verdict=image_verdict,
                        fraud_risk_band=fraud_risk_band,
                        fraud_key_signals=["total_returns_lifetime"] if fraud_risk_band != "low" else [],
                    )
                )

    for fraud_risk_band in _FRAUD_BANDS:
        scenarios.append(
            GoldenScenario(
                scenario_id=f"change_of_mind_{fraud_risk_band}",
                claim_category="Change of Mind",
                claim_description=_DESCRIPTIONS["Change of Mind"],
                refund_amount_usd=60.0,
                image_verdict="consistent",
                fraud_risk_band=fraud_risk_band,
            )
        )

    # Explicit >$200 guardrail-override cases -- same cells as above, but a
    # high refund amount that must force 'escalate' regardless of what the
    # matrix cell alone would say (the guardrail is checked first).
    for image_verdict, fraud_risk_band, category in [
        ("consistent", "low", "Damaged in Transit"),  # matrix alone says approve
        ("consistent", "medium", "Wrong Item Received"),  # matrix alone says approve
        ("partially_consistent", "low", "Not as Described"),  # matrix alone says approve
        ("inconsistent", "medium", "Defective/DOA"),  # matrix alone says deny
    ]:
        scenarios.append(
            GoldenScenario(
                scenario_id=f"guardrail_override_{image_verdict}_{fraud_risk_band}",
                claim_category=category,
                claim_description=_DESCRIPTIONS[category],
                refund_amount_usd=350.0,
                image_verdict=image_verdict,
                fraud_risk_band=fraud_risk_band,
            )
        )

    return scenarios


GOLDEN_SET = build_golden_set()
