"""Real agent-invocation helpers shared by test_ner_pii_leak.py and
similarity_entity_report.py -- both need the same real calls (a Decision
Agent verdict's reasoning, an Image Parsing Agent's consistency
reasoning), so the actual Crew-running logic lives here once rather than
duplicated between a gating test and an informational report.

Image Parsing helper re-added here (project-plan.md Q87) after being
pulled in Q86 -- a real, confirmed base64-truncation bug made every real
Image Parsing scenario fail the same way at redact_photo; the fix (Q86/
Q87's analyze_claim_photo, a single server-side tool the agent calls with
just claim_ref/order_ref/claim_category/claim_description, never handling
the raw photo bytes itself) landed, so this is no longer demonstrating a
known-broken path.
"""

import base64
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agents.decision_agent import build_decision_agent, build_verdict_task  # noqa: E402
from agents.image_parsing_agent import build_consistency_task, build_image_parsing_agent  # noqa: E402
from agents.mcp_tools import decision_mcp_adapter, image_parsing_mcp_adapter, orchestrator_mcp_adapter  # noqa: E402

from crewai import Crew  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_set import GOLDEN_SET  # noqa: E402


def run_decision_reasoning(scenario_id: str) -> str:
    """A real Decision Agent verdict for one golden_set.py scenario --
    returns just the real reasoning text this module's metrics score."""
    scenario = next(s for s in GOLDEN_SET if s.scenario_id == scenario_id)
    with decision_mcp_adapter() as tools:
        agent = build_decision_agent(tools, customer_ref="cst_ner_eval", claim_ref=scenario.claim_ref)
        task = build_verdict_task(
            agent, claim_ref=scenario.claim_ref, order_ref="1", claim_category=scenario.claim_category,
            claim_description=scenario.claim_description, refund_amount_usd=scenario.refund_amount_usd,
            image_verdict=scenario.image_verdict, fraud_risk_band=scenario.fraud_risk_band,
            fraud_key_signals=scenario.fraud_key_signals,
        )
        result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
    return result.pydantic.reasoning


def run_image_parsing_reasoning(reference_scenario) -> str:
    """A real Image Parsing Agent consistency verdict for one
    reference_answers.py ImageParsingReferenceScenario -- stores the real
    photo via the Orchestrator's store_photo first (store_photo isn't
    Orchestrator-restricted the way issue_refund is, so calling it from a
    different MCP client session than the one that later reads it back is
    fine, the same pattern backend/main.py's photo endpoint uses), then
    runs the real Image Parsing Agent, which now only ever calls
    analyze_claim_photo with claim_ref -- it never sees the photo bytes."""
    claim_ref = f"clm_ner_eval_{uuid.uuid4().hex[:8]}"
    photo_base64 = base64.b64encode(reference_scenario.photo_path.read_bytes()).decode("ascii")
    with orchestrator_mcp_adapter() as tools:
        store_photo = next(t for t in tools if t.name == "store_photo")
        store_photo.run(claim_ref=claim_ref, photo_base64=photo_base64)

    with image_parsing_mcp_adapter() as tools:
        agent = build_image_parsing_agent(tools, customer_ref="cst_ner_eval", claim_ref=claim_ref)
        task = build_consistency_task(
            agent, claim_ref=claim_ref, order_ref=reference_scenario.order_ref,
            claim_category=reference_scenario.claim_category, claim_description=reference_scenario.claim_description,
        )
        result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
    return result.pydantic.reasoning
