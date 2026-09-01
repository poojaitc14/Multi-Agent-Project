"""Property tests (testing-and-evaluation-plan.md's "Eval harness design"):
randomized/generated inputs checked against invariants that must hold
regardless of the specific values, rather than one exact expected verdict
per scenario (that's golden_set.py's job). Real, not mocked.

Two of the plan's four listed invariants are implemented here:
  - refund_amount_usd > $200 always escalates, regardless of image/fraud
  - a photo-required category with no stored photo always re-prompts
    rather than producing a Verdict

Two are deliberately not yet implemented, flagged rather than faked:
  - "issue_refund is only ever called after an approve" -- already a real,
    structural property proven by tests/test_security_and_pii.py's
    test_issue_refund_not_offered_to_any_non_orchestrator_agent plus
    tests/test_orchestrator_server.py's token-gate tests; not duplicated
    here as a "property test" in this file's narrower sense.
  - "a search_refund_policy call with no confidently relevant chunk always
    escalates" -- needs a real scenario that reliably forces the Decision
    Agent's own generated query into low-confidence territory, which isn't
    a straightforward randomized input the way the other two are; left as
    a real, open gap for a future pass rather than simulated.
"""

import random
import sys
import uuid
from pathlib import Path

import pytest
from crewai import Crew

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agents.decision_agent import build_decision_agent, build_verdict_task  # noqa: E402
from agents.flow import ClaimTriageFlow  # noqa: E402
from agents.mcp_tools import decision_mcp_adapter, orchestrator_mcp_adapter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_set import _IMAGE_VERDICTS as IMAGE_VERDICTS  # noqa: E402
from golden_set import _FRAUD_BANDS as FRAUD_BANDS  # noqa: E402


@pytest.mark.parametrize("seed", range(5))
def test_refund_amount_over_200_always_escalates_real_tool(seed):
    """Real apply_decision_matrix calls, randomized (image_verdict,
    fraud_risk_band) each run -- the invariant must hold for every
    combination, not just the ones golden_set.py happens to enumerate."""
    rng = random.Random(seed)
    image_verdict = rng.choice(IMAGE_VERDICTS)
    fraud_risk_band = rng.choice(FRAUD_BANDS)
    refund_amount_usd = rng.uniform(200.01, 5000.0)

    with decision_mcp_adapter() as tools:
        apply_decision_matrix = next(t for t in tools if t.name == "apply_decision_matrix")
        result = apply_decision_matrix.run(
            image_verdict=image_verdict, fraud_risk_band=fraud_risk_band, refund_amount_usd=refund_amount_usd,
        )
    import json

    decision = json.loads(result)["decision"] if isinstance(result, str) else result["decision"]
    assert decision == "escalate", (
        f"${refund_amount_usd:.2f} with image_verdict={image_verdict!r}, "
        f"fraud_risk_band={fraud_risk_band!r} did not escalate: got {decision!r}"
    )


@pytest.mark.parametrize("seed", range(3))
def test_refund_amount_over_200_always_escalates_real_agent(seed):
    """Same invariant, through the real Decision Agent's own reported
    decision (not just the tool it's supposed to defer to) -- confirms the
    LLM never overrides the guardrail's real result even when asked to
    reason about a case where the underlying matrix cell alone would say
    something else."""
    rng = random.Random(1000 + seed)
    image_verdict = rng.choice(IMAGE_VERDICTS)
    fraud_risk_band = rng.choice(FRAUD_BANDS)
    refund_amount_usd = rng.uniform(200.01, 2000.0)
    claim_ref = f"clm_property_{uuid.uuid4().hex[:8]}"

    with decision_mcp_adapter() as tools:
        agent = build_decision_agent(tools, customer_ref="cst_eval", claim_ref=claim_ref)
        task = build_verdict_task(
            agent, claim_ref=claim_ref, order_ref="1", claim_category="Damaged in Transit",
            claim_description="Item arrived damaged.", refund_amount_usd=refund_amount_usd,
            image_verdict=image_verdict, fraud_risk_band=fraud_risk_band, fraud_key_signals=[],
        )
        result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()

    assert result.pydantic.decision == "escalate"


def test_no_photo_on_photo_required_category_always_reprompts():
    """A real, full ClaimTriageFlow run (not just the Decision Agent) --
    Damaged in Transit requires a photo (agents/flow.py's
    PHOTO_REQUIRED_CATEGORIES); no photo has been stored for this
    brand-new claim_ref, so Image Parsing must genuinely find none and the
    Flow must stop at re_prompt_for_photo without ever reaching Fraud
    Scoring or the Decision Agent (project-plan.md Q65 -- already proven
    once in Slice 9's manual verification; this makes it a repeatable
    regression test instead of a one-off)."""
    with orchestrator_mcp_adapter() as tools:
        resolve = next(t for t in tools if t.name == "resolve_customer_ref")
        customer_ref = resolve.run(customer_id_or_email=f"property-test-{uuid.uuid4().hex[:8]}@example.com")

    flow = ClaimTriageFlow()
    flow.kickoff(
        inputs={
            "claim_ref": f"clm_property_{uuid.uuid4().hex[:8]}",
            "customer_identifier": customer_ref,
            "order_ref": "1",
            "claim_category": "Damaged in Transit",
            "claim_description": "Item arrived damaged, no photo taken.",
            "days_to_return": 3,
        }
    )

    assert flow.state.outcome == "re_prompt_for_photo"
    assert flow.state.fraud_risk_band == "", "Fraud Scoring must never run on a re-prompt"
    assert flow.state.decision == "", "the Decision Agent must never run on a re-prompt"
