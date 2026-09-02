"""Orchestrator Agent (project-plan.md's "Agents & models" + "Tool List Per
Agent"). Its own LLM-driven job is narrow: classify the incoming message
into a ClassificationResult. Everything else the plan describes as the
Orchestrator's job -- resolving customer_ref, invoking Image Parsing +
Fraud Scoring in parallel, handing both to the Decision Agent, applying the verdict via
issue_refund -- is deterministic Flow-level control code (agents/flow.py),
not something delegated to this agent's own judgment. issue_refund in
particular is never called because an LLM decided to; it's called by Flow
code gated on the Decision Agent's real Verdict.decision, per the "only caller of
issue_refund" / non-bypassable-write-path design.
"""

from typing import Any, Optional, Tuple

from crewai import Agent, Task
from crewai.tasks.task_output import TaskOutput

from .llms import get_gpt41_mini_llm
from .schemas import ClaimIntakeResult, ClassificationResult

REQUIRED_FOR_NEW_CLAIM = ("claim_category", "claim_description")


def build_orchestrator_agent(tools, customer_ref: Optional[str] = None, claim_ref: Optional[str] = None) -> Agent:
    """customer_ref/claim_ref (Q30/Q73): forwarded to get_gpt41_mini_llm so
    this agent's real LLM calls get tagged for Langfuse tracing -- never
    the raw customer identifier, only the already-resolved ref."""
    return Agent(
        role="Orchestrator",
        goal=(
            "Classify an incoming customer message as a new return/refund claim, a "
            "follow-up on an existing claim, or a general inquiry -- and extract the "
            "claim category and description."
        ),
        backstory=(
            "You triage incoming customer messages for an e-commerce returns system. "
            "Which order a claim is about is decided entirely outside your own judgment "
            "-- the customer picks it from a real, verified dropdown of their own orders "
            "before you ever see their message (project-plan.md Q101). You never try to "
            "extract, infer, or override an order reference yourself, even if the "
            "customer's message happens to mention a different order number -- that's "
            "not your call to make."
        ),
        llm=get_gpt41_mini_llm(customer_ref, claim_ref),
        tools=tools,
        verbose=True,
    )


def classification_guardrail(output: TaskOutput) -> Tuple[bool, Any]:
    """Layer 3: a business rule schema validation alone can't express -- a
    new_claim or follow_up without an order_ref isn't actionable downstream,
    even though ClassificationResult's schema allows order_ref to be None
    (general_inquiry legitimately has none)."""
    result = output.pydantic
    if result is None:
        return False, "expected a ClassificationResult, got no structured output"
    if result.request_type in ("new_claim", "follow_up") and not result.order_ref:
        return False, f"request_type={result.request_type!r} requires an order_ref, but none was extracted"
    return True, result


def build_classification_task(agent: Agent, customer_message: str) -> Task:
    return Task(
        description=(
            f'A customer sent this message: "{customer_message}"\n\n'
            "Classify it as one of: new_claim (reporting a new issue with an order "
            "for the first time), follow_up (referencing a claim already in "
            "progress), or general_inquiry (no specific claim, e.g. a policy "
            "question). Extract the order reference if the message states or "
            "clearly implies one; otherwise leave it unset."
        ),
        expected_output="A ClassificationResult with request_type and, when applicable, order_ref.",
        agent=agent,
        output_pydantic=ClassificationResult,
        guardrail=classification_guardrail,
    )


def intake_guardrail(output: TaskOutput) -> Tuple[bool, Any]:
    """Layer 3, checked in code rather than trusted to the LLM's own
    self-report: a new_claim/follow_up with any of REQUIRED_FOR_NEW_CLAIM
    still missing must carry a real follow_up_question (the whole point of
    multi-turn intake); conversely, once every field is genuinely present,
    asking a follow_up_question anyway would leave a complete claim stuck
    in an unnecessary extra round-trip."""
    result = output.pydantic
    if result is None:
        return False, "expected a ClaimIntakeResult, got no structured output"
    if result.request_type in ("new_claim", "follow_up"):
        missing = [f for f in REQUIRED_FOR_NEW_CLAIM if getattr(result, f) is None]
        if missing and not result.follow_up_question:
            return False, f"fields {missing} are still missing but no follow_up_question was asked"
        if not missing and result.follow_up_question:
            return False, "every required field is present -- no follow_up_question should be asked"
    return True, result


def build_intake_task(agent: Agent, customer_message: str, known_fields: dict) -> Task:
    """known_fields: whatever was already extracted from earlier messages
    in this same claim's conversation (from get_conversation_state) --
    empty dict on a genuinely first message. Passed as context so the
    agent doesn't need to re-ask for something already established; Flow-
    level code (backend/main.py) still merges the result with known_fields
    afterward as a safety net, rather than trusting the agent to always
    carry every prior field forward into its own output.

    order_ref/days_to_return (project-plan.md Q101): deliberately not part
    of what this task asks the LLM to extract at all anymore. order_ref
    must come from the Customer Chat Frontend's real order dropdown (Q99)
    -- backend/main.py enforces this deterministically and never accepts
    whatever value the LLM's own ClaimIntakeResult.order_ref happens to
    hold, so there's no reason to prompt for it here. days_to_return is
    now computed server-side from that order's real, bootstrap-seeded
    order_date (Q98/Q101), not self-reported -- a real fraud-detection gap
    this closes: a customer's own estimate of "how many days ago" was
    previously never checked against anything real."""
    known_summary = ", ".join(f"{k}={v!r}" for k, v in known_fields.items() if v is not None) or "nothing yet"
    return Task(
        description=(
            f'A customer sent this message: "{customer_message}"\n\n'
            f"Already known from earlier in this conversation: {known_summary}.\n\n"
            "Classify request_type (new_claim / follow_up / general_inquiry). For a "
            "new_claim or follow_up, extract as many of these as the message provides: "
            "claim_category (must be exactly one of: 'Damaged in Transit', "
            "'Wrong Item Received', 'Not as Described', 'Defective/DOA', 'Change of Mind'), "
            "claim_description. Never invent a value that isn't stated or clearly implied "
            "-- leave a field unset if it's genuinely not known yet. Do not extract or "
            "report an order_ref or days_to_return -- which order this is about is decided "
            "entirely by which order the customer picked from the real dropdown above the "
            "message box, never from anything stated in the message text itself. If either "
            "claim_category or claim_description remains unknown after this message, write "
            "a short, natural follow_up_question asking the customer for exactly what's "
            "still missing. If both are now known, leave follow_up_question unset."
        ),
        expected_output="A ClaimIntakeResult with claim_category/claim_description (order_ref and days_to_return left unset), and a follow_up_question if anything is still missing.",
        agent=agent,
        output_pydantic=ClaimIntakeResult,
        guardrail=intake_guardrail,
    )
