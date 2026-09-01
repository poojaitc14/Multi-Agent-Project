"""ClaimTriageFlow -- the real CrewAI Flow wiring all 4 agents together
(project-plan.md's "Communication" section, revised by Q62/Q65).

Orchestrator resolves the customer_ref and order details (deterministic,
non-LLM MCP calls -- Orchestrator's own Agent/Task isn't invoked here,
since claim intake in this slice is already-structured input, not a raw
customer message needing classification; see the module docstring's
"Not yet built" note). Then:

  resolve_customer_and_order
      -> [route_photo_requirement]
           -> run_image_parsing (photo-required categories, Q65)
           -> use_default_image_consistency (Change of Mind, Q65 -- Image
              Parsing is skipped entirely, not run-and-ignored)
      -> [route_after_image_parsing]
           -> handle_image_analysis_unavailable (Image Parsing's guardrail
              exhausted retries on real, repeated base64-corruption failures
              -- proven to happen for real, not hypothetical -- degrades to
              human review rather than crashing the Flow)
           -> handle_no_photo_reprompt (STOPS here -- Fraud Scoring and
              Decision never run, Q65: "no verdict yet" is not a Verdict)
           -> run_fraud_scoring (image_verdict comes from whichever path above ran)
      -> run_decision  (apply_decision_matrix grounds the label; may still
                         fail honestly on a real API/guardrail error --
                         caught and surfaced as outcome='decision_unavailable',
                         never faked)
      -> [route_after_decision]
           -> do_issue_refund (only ever on a genuine "approve")
           -> finalize_non_approve (deny/escalate)
           -> finalize_decision_unavailable

issue_refund is called via the Orchestrator's own MCP client/token
(ORCHESTRATOR_MCP_TOKEN) -- Flow-level code, not an agent's own judgment
call, matching "the only caller of issue_refund" design (Q11) exactly the
way score_fraud_risk's risk_band and apply_decision_matrix's decision are
grounded in real logic rather than trusted to an LLM.
"""

import json
from typing import Any, Optional

from crewai import Crew
from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel

from .decision_agent import build_decision_agent, build_verdict_task
from .fraud_scoring_agent import build_fraud_scoring_agent, build_fraud_scoring_task
from .image_parsing_agent import build_consistency_task, build_image_parsing_agent
from .mcp_tools import (
    decision_mcp_adapter,
    fraud_scoring_mcp_adapter,
    image_parsing_mcp_adapter,
    orchestrator_mcp_adapter,
)

# project-plan.md's "Claim categories" (Refund policy section) -- all but
# Change of Mind require a photo (Q65).
PHOTO_REQUIRED_CATEGORIES = {
    "Damaged in Transit",
    "Wrong Item Received",
    "Not as Described",
    "Defective/DOA",
}


def _parse_tool_result(result: Any) -> Any:
    """crewai_tools' MCPServerAdapter Tool.run() JSON-stringifies a dict/
    object MCP result (confirmed empirically: get_order and issue_refund
    both come back as JSON strings, not dicts), but leaves an
    already-plain-string MCP result (e.g. resolve_customer_ref) untouched.
    Only parses if it looks like it needs it, so a real plain string never
    gets mis-parsed."""
    if isinstance(result, str) and result.strip().startswith(("{", "[")):
        return json.loads(result)
    return result


class ClaimState(BaseModel):
    # Claim intake (already-structured input to this Flow -- see module docstring)
    claim_ref: str = ""
    customer_identifier: str = ""
    order_ref: str = ""
    claim_category: str = ""
    claim_description: str = ""
    days_to_return: int = 0

    # Resolved during the flow
    customer_ref: str = ""
    refund_amount_usd: float = 0.0

    # Image Parsing output
    image_verdict: str = ""
    product_match: bool = False
    image_reasoning: str = ""
    photo_evidence_provided: bool = False
    image_analysis_error: Optional[str] = None

    # Fraud Scoring output
    fraud_risk_band: str = ""
    fraud_risk_score: float = 0.0
    fraud_key_signals: list = []
    fraud_reasoning: str = ""

    # Decision output
    decision: str = ""
    refund_form: Optional[str] = None
    policy_clause: Optional[str] = None
    policy_version: Optional[str] = None
    verdict_reasoning: str = ""
    decision_error: Optional[str] = None

    # Final outcome
    outcome: str = ""
    transaction_id: Optional[str] = None


class ClaimTriageFlow(Flow[ClaimState]):
    @start()
    def resolve_customer_and_order(self) -> str:
        with orchestrator_mcp_adapter() as tools:
            resolve = next(t for t in tools if t.name == "resolve_customer_ref")
            get_order = next(t for t in tools if t.name == "get_order")
            self.state.customer_ref = resolve.run(customer_id_or_email=self.state.customer_identifier)
            order = _parse_tool_result(get_order.run(order_ref=self.state.order_ref))
            self.state.refund_amount_usd = float(order["amount"])
        return self.state.customer_ref

    @router(resolve_customer_and_order)
    def route_photo_requirement(self) -> str:
        if self.state.claim_category in PHOTO_REQUIRED_CATEGORIES:
            return "run_image_parsing"
        return "skip_image_parsing"

    @listen("skip_image_parsing")
    def use_default_image_consistency(self) -> str:
        """Q65: Image Parsing is skipped entirely for Change of Mind, not
        run and ignored -- no GPT-4.1 mini vision call happens for a claim
        that will never have a photo. 'consistent' is a neutral default,
        not a real judgment -- there is nothing to judge."""
        self.state.image_verdict = "consistent"
        self.state.photo_evidence_provided = False
        return self.state.image_verdict

    @listen("run_image_parsing")
    def do_image_parsing(self) -> str:
        """May still exhaust consistency_guardrail's retries and raise (a
        real, genuine analyze_claim_photo failure -- Azure vision API down,
        a corrupted stored photo, etc.) -- caught here, same graceful-
        degradation shape as do_decision, rather than crashing the whole
        Flow. This used to also catch a specific, confirmed base64-
        truncation bug in the agent's own tool-call argument generation
        (project-plan.md Q86): a 6KB test photo hit it on 4/4 attempts in
        one real run, later confirmed at 100% reproducibility and root-
        caused as long-string generation drift. That's fixed now (Q87) --
        the photo bytes never enter the LLM's own generated arguments at
        all anymore, see image_parsing_agent.py -- so this except clause is
        back to handling genuine, unrelated failures, not routinely
        catching that specific one."""
        try:
            with image_parsing_mcp_adapter() as tools:
                agent = build_image_parsing_agent(tools, self.state.customer_ref, self.state.claim_ref)
                task = build_consistency_task(
                    agent, self.state.claim_ref, self.state.order_ref,
                    self.state.claim_category, self.state.claim_description,
                )
                result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
            verdict = result.pydantic
            self.state.image_verdict = verdict.verdict
            self.state.product_match = verdict.product_match
            self.state.image_reasoning = verdict.reasoning
            self.state.photo_evidence_provided = verdict.verdict != "no_photo"
            return self.state.image_verdict
        except Exception as e:  # noqa: BLE001 -- deliberately broad: any Image Parsing failure degrades honestly
            self.state.image_analysis_error = f"{type(e).__name__}: {e}"
            return "image_analysis_unavailable"

    @router(do_image_parsing)
    def route_after_image_parsing(self) -> str:
        if self.state.image_analysis_error:
            return "image_analysis_unavailable"
        if self.state.image_verdict == "no_photo":
            return "no_photo_reprompt"
        return "run_fraud_scoring"

    @listen("image_analysis_unavailable")
    def handle_image_analysis_unavailable(self) -> ClaimState:
        self.state.outcome = "image_analysis_unavailable"
        return self.state

    @listen("no_photo_reprompt")
    def handle_no_photo_reprompt(self) -> ClaimState:
        """Q65: stops here -- Fraud Scoring and the Decision Agent never
        run. The decision matrix's 'no verdict yet' isn't a Verdict."""
        self.state.outcome = "re_prompt_for_photo"
        return self.state

    @listen(or_("run_fraud_scoring", use_default_image_consistency))
    def do_fraud_scoring(self) -> str:
        with fraud_scoring_mcp_adapter() as tools:
            agent = build_fraud_scoring_agent(tools, self.state.customer_ref, self.state.claim_ref)
            task = build_fraud_scoring_task(
                agent,
                customer_ref=self.state.customer_ref,
                order_ref=self.state.order_ref,
                claim_ref=self.state.claim_ref,
                claim_category=self.state.claim_category,
                claim_description=self.state.claim_description,
                days_to_return=self.state.days_to_return,
                photo_evidence_provided=self.state.photo_evidence_provided,
                image_consistency=self.state.image_verdict,
            )
            result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        assessment = result.pydantic
        self.state.fraud_risk_band = assessment.risk_band
        self.state.fraud_risk_score = assessment.risk_score
        self.state.fraud_key_signals = assessment.key_signals
        self.state.fraud_reasoning = assessment.reasoning
        return self.state.fraud_risk_band

    @listen(do_fraud_scoring)
    def do_decision(self) -> str:
        """May still fail honestly on a real API or guardrail-exhaustion
        error (Q52/Q75) -- caught here rather than crashing the whole Flow,
        same graceful-degradation shape as do_image_parsing."""
        try:
            with decision_mcp_adapter() as tools:
                agent = build_decision_agent(tools, self.state.customer_ref, self.state.claim_ref)
                task = build_verdict_task(
                    agent,
                    claim_ref=self.state.claim_ref,
                    order_ref=self.state.order_ref,
                    claim_category=self.state.claim_category,
                    claim_description=self.state.claim_description,
                    refund_amount_usd=self.state.refund_amount_usd,
                    image_verdict=self.state.image_verdict,
                    fraud_risk_band=self.state.fraud_risk_band,
                    fraud_key_signals=self.state.fraud_key_signals,
                )
                result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
            verdict = result.pydantic
            self.state.decision = verdict.decision
            self.state.refund_form = verdict.refund_form
            self.state.policy_clause = verdict.policy_clause
            self.state.policy_version = verdict.policy_version
            self.state.verdict_reasoning = verdict.reasoning
            return "decision_made"
        except Exception as e:  # noqa: BLE001 -- deliberately broad: any Decision Agent failure degrades honestly
            self.state.decision_error = f"{type(e).__name__}: {e}"
            return "decision_unavailable"

    @router(do_decision)
    def route_after_decision(self) -> str:
        if self.state.decision_error:
            return "decision_unavailable"
        if self.state.decision == "approve":
            return "approved"
        return "deny_or_escalate"

    @listen("approved")
    def do_issue_refund(self) -> ClaimState:
        """The only place issue_refund is ever called -- gated on a real
        Verdict.decision == 'approve' already checked above, called with
        the Orchestrator's own token, never an agent's own tool call
        (Q11's non-bypassable write path)."""
        with orchestrator_mcp_adapter() as tools:
            issue_refund = next(t for t in tools if t.name == "issue_refund")
            result = _parse_tool_result(
                issue_refund.run(
                    order_ref=self.state.order_ref,
                    claim_ref=self.state.claim_ref,
                    amount=self.state.refund_amount_usd,
                    reason=f"{self.state.claim_category}: {self.state.claim_description}",
                )
            )
        self.state.transaction_id = result.get("transaction_id")
        self.state.outcome = "refunded"
        return self.state

    @listen("deny_or_escalate")
    def finalize_non_approve(self) -> ClaimState:
        self.state.outcome = self.state.decision  # "deny" or "escalate"
        return self.state

    @listen("decision_unavailable")
    def finalize_decision_unavailable(self) -> ClaimState:
        self.state.outcome = "decision_unavailable"
        return self.state
