"""The golden-set eval harness (testing-and-evaluation-plan.md's "Eval
harness design" + the "Tool routing accuracy" and "Tool call sequence"
rows of "Agent evaluation metrics"). Real, not mocked: every scenario
makes a genuine Decision Agent LLM call (GPT-4.1 mini, temperature=0 per
Q6) through the real MCP server.

NOT part of the fast, per-push CI gate (testing-and-evaluation-plan.md's
"CI gate policy", Q12) -- 36 real LLM calls is slow and, unlike this
project's other pytest suites, deliberately allowed to be. Run this file
on a schedule / immediately before deploy, same as the plan specifies,
not on every push.

Scope, and what's deliberately NOT covered here yet (see
testing-and-evaluation-plan.md's remaining open items and
project-plan.md's eval-harness slice for why): semantic similarity and
named-entity-recognition metrics need a hosted embeddings/NER service,
which Q17 flagged as needing the same PII review GPT-4.1 mini went
through before a specific provider is chosen -- not decided here, so
those two metrics aren't implemented yet. RAGAS (this file's natural next
extension, now that real OpenSearch retrieval works per project-plan.md
Q81) and cost-per-claim tracking (via Langfuse, already wired) are also
not yet implemented -- flagged as the next real increment, not silently
skipped.
"""

import sys
from pathlib import Path

import pytest
from crewai import Crew

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agents.decision_agent import build_decision_agent, build_verdict_task  # noqa: E402
from agents.mcp_tools import decision_mcp_adapter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_set import EXPECTED_DECISION_AGENT_TOOLS, GOLDEN_SET, expected_decision, tool_call_sequence  # noqa: E402


@pytest.mark.parametrize("scenario", GOLDEN_SET, ids=lambda s: s.scenario_id)
def test_golden_scenario_matches_expected_decision(scenario):
    """The core golden-set check: does the real Decision Agent report
    exactly the decision docs/refund_policy.md's matrix + guardrail says
    it should, for a known (image_verdict, fraud_risk_band,
    refund_amount_usd) triple. verdict_guardrail (agents/decision_agent.py)
    already prevents the agent from reporting anything OTHER than
    apply_decision_matrix's real result -- this test is the outer,
    end-to-end confirmation that the whole chain (real tool call -> real
    guardrail -> real reported verdict) lands on the objectively correct
    label, not just "some grounded label"."""
    with decision_mcp_adapter() as tools:
        agent = build_decision_agent(tools, customer_ref="cst_eval", claim_ref=scenario.claim_ref)
        task = build_verdict_task(
            agent,
            claim_ref=scenario.claim_ref,
            order_ref="1",
            claim_category=scenario.claim_category,
            claim_description=scenario.claim_description,
            refund_amount_usd=scenario.refund_amount_usd,
            image_verdict=scenario.image_verdict,
            fraud_risk_band=scenario.fraud_risk_band,
            fraud_key_signals=scenario.fraud_key_signals,
        )
        result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()

    verdict = result.pydantic
    assert verdict.decision == scenario.expected_decision, (
        f"scenario {scenario.scenario_id}: expected {scenario.expected_decision!r}, "
        f"agent reported {verdict.decision!r}"
    )

    # Tool routing accuracy (testing-and-evaluation-plan.md Q9): only the
    # Decision Agent's own two real tools, nothing outside its MCP server.
    messages = result.tasks_output[0].messages
    called_tools = set(tool_call_sequence(messages))
    assert called_tools <= EXPECTED_DECISION_AGENT_TOOLS, (
        f"scenario {scenario.scenario_id}: called unexpected tool(s) {called_tools - EXPECTED_DECISION_AGENT_TOOLS}"
    )
    assert "apply_decision_matrix" in called_tools, f"scenario {scenario.scenario_id}: never called apply_decision_matrix"

    # Tool call sequence (testing-and-evaluation-plan.md Q9): search_refund_policy
    # before apply_decision_matrix -- retrieval must inform the verdict, not
    # happen incidentally after it (or not at all).
    sequence = tool_call_sequence(messages)
    if "search_refund_policy" in sequence and "apply_decision_matrix" in sequence:
        assert sequence.index("search_refund_policy") < sequence.index("apply_decision_matrix"), (
            f"scenario {scenario.scenario_id}: apply_decision_matrix called before search_refund_policy"
        )


@pytest.mark.parametrize(
    "image_verdict,fraud_risk_band,refund_amount_usd",
    [
        ("consistent", "low", 50.0), ("consistent", "medium", 50.0), ("consistent", "high", 50.0),
        ("partially_consistent", "low", 50.0), ("partially_consistent", "medium", 50.0), ("partially_consistent", "high", 50.0),
        ("inconsistent", "low", 50.0), ("inconsistent", "medium", 50.0), ("inconsistent", "high", 50.0),
        ("consistent", "low", 250.0),  # guardrail override case
    ],
)
def test_mirrored_decision_matrix_matches_the_real_tool(image_verdict, fraud_risk_band, refund_amount_usd):
    """golden_set.py's _DECISION_MATRIX/_HIGH_VALUE_ESCALATION_THRESHOLD_USD
    are a literal mirror of mcp-servers/orchestrator_server.py's real
    constants, not a re-derivation -- this is the regression guard that
    catches drift between the two, rather than trusting the mirror
    silently forever. Not part of the golden-set scenarios above (those
    test the Decision Agent's LLM behavior); this tests the real
    apply_decision_matrix MCP tool directly, no LLM involved."""
    with decision_mcp_adapter() as tools:
        apply_decision_matrix = next(t for t in tools if t.name == "apply_decision_matrix")
        real_result = apply_decision_matrix.run(
            image_verdict=image_verdict, fraud_risk_band=fraud_risk_band, refund_amount_usd=refund_amount_usd,
        )
    import json

    real_decision = json.loads(real_result)["decision"] if isinstance(real_result, str) else real_result["decision"]
    assert real_decision == expected_decision(image_verdict, fraud_risk_band, refund_amount_usd)
