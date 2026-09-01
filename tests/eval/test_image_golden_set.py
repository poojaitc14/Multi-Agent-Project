"""Runs image_golden_set.py's 25 real scenarios through the real Image
Parsing Agent (same testing philosophy as test_golden_set.py -- through
the actual CrewAI agent over the live MCP server, not a raw tool call),
confirming project-plan.md Q88's analyze_claim_photo fix holds across a
real, varied 25-scenario set, not just the original 3-scenario sample.

Every scenario is a real pytest test (must produce a well-formed, grounded
ConsistencyAssessment -- consistency_guardrail already enforces that,
catching fabrication/crashes for all 25). The 9 hard-gated scenarios
additionally assert their genuinely knowable expected_verdict/
expected_product_match; the 16 informational scenarios print their real
result for visibility instead of asserting a ground truth this file has
no honest way to know in advance (see image_golden_set.py's module
docstring for why that split exists).

Run with: uv run pytest tests/eval/test_image_golden_set.py -v -s
(the -s flag surfaces the informational-scenario print output, which
pytest normally captures/hides on a pass)
"""

import base64
import sys
from pathlib import Path

import pytest
from crewai import Crew

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agents.image_parsing_agent import build_consistency_task, build_image_parsing_agent  # noqa: E402
from agents.mcp_tools import image_parsing_mcp_adapter, orchestrator_mcp_adapter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_golden_set import IMAGE_GOLDEN_SET  # noqa: E402


def _run_scenario(scenario):
    if scenario.photo_path is not None:
        photo_base64 = base64.b64encode(scenario.photo_path.read_bytes()).decode("ascii")
        with orchestrator_mcp_adapter() as tools:
            store_photo = next(t for t in tools if t.name == "store_photo")
            store_photo.run(claim_ref=scenario.claim_ref, photo_base64=photo_base64)

    with image_parsing_mcp_adapter() as tools:
        agent = build_image_parsing_agent(tools, customer_ref="cst_img_golden", claim_ref=scenario.claim_ref)
        task = build_consistency_task(
            agent, claim_ref=scenario.claim_ref, order_ref=scenario.order_ref,
            claim_category=scenario.claim_category, claim_description=scenario.claim_description,
        )
        result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
    return result.pydantic


@pytest.mark.parametrize("scenario", IMAGE_GOLDEN_SET, ids=lambda s: s.scenario_id)
def test_image_golden_scenario(scenario):
    assessment = _run_scenario(scenario)

    assert assessment.verdict in ("consistent", "partially_consistent", "inconsistent", "no_photo")
    assert isinstance(assessment.product_match, bool)
    assert isinstance(assessment.reasoning, str) and len(assessment.reasoning) > 0

    if scenario.expected_verdict is not None:
        assert assessment.verdict == scenario.expected_verdict, (
            f"{scenario.scenario_id}: expected verdict={scenario.expected_verdict!r}, got {assessment.verdict!r}"
        )
    if scenario.expected_product_match is not None:
        assert assessment.product_match == scenario.expected_product_match, (
            f"{scenario.scenario_id}: expected product_match={scenario.expected_product_match!r}, "
            f"got {assessment.product_match!r}"
        )

    if not scenario.is_hard_gated:
        print(
            f"\n[informational] {scenario.scenario_id}: verdict={assessment.verdict!r} "
            f"product_match={assessment.product_match!r}"
        )
