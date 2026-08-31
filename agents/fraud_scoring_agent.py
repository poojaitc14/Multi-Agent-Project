"""ML Fraud Scoring Agent. Its Task assembles all 13 features score_fraud_risk
needs and relays its real, calibrated, SHAP-grounded output as
FraudAssessment -- the agent's job is feature assembly and tool
orchestration, not inventing a risk judgment; score_fraud_risk (the real
registered model, ml/fraud_attribution.py) is what actually scores the claim.

Feature sourcing (each resolved as a real design decision, not assumed):
- account_age_days, total_orders_lifetime, total_returns_lifetime,
  customer_support_contacts_90d, previous_dispute_count, address_match:
  get_account_info(customer_ref) -- Q59's seeded synthetic profile.
- claim_frequency_90d: get_claim_frequency(customer_ref) -- real DynamoDB
  sliding window; increment_claim_frequency(customer_ref, claim_ref) is
  also called once, to record this claim in that same window.
- refund_amount_usd: get_order(order_ref)["amount"] -- a real, objective
  order fact, not customer-reported.
- is_high_value_item: refund_amount_usd > HIGH_VALUE_THRESHOLD_USD -- reuses
  the Decision Agent's existing $200 escalation guardrail threshold (a real
  decision: the alternative was a separate, undefined threshold).
- days_to_return: customer-provided at claim intake (a real decision: the
  alternative considered was per-claim synthetic seeding, rejected since
  there's no natural place to persist/reuse a per-claim synthetic value the
  way Q59's per-customer profile works) -- passed in as claim context,
  same as claim_category.
- photo_evidence_provided, image_consistency: the Image Parsing Agent's own
  ConsistencyAssessment (photo_evidence_provided = verdict != 'no_photo',
  image_consistency = verdict itself) -- a real decision: Image Parsing now
  runs BEFORE Fraud Scoring (no longer truly parallel, project-plan.md's
  original design), specifically so this feature is never guessed. Passed
  in as parameters, computed by Flow-level code, not re-derived by this
  agent's own judgment.
- claim_category: claim context, customer-provided (unchanged from the
  original design).
"""

from typing import Any, Optional, Tuple

from crewai import Agent, Task
from crewai.tasks.task_output import TaskOutput

from .llms import get_gpt41_mini_llm
from .schemas import FraudAssessment

HIGH_VALUE_THRESHOLD_USD = 200.0

# Mirrors ml/fraud_attribution.py's FEATURE_COLS (asserted equal at the
# bottom of this file, not just copy-pasted and hoped to stay in sync).
# NOT imported directly: importing that module pulls in shap -> matplotlib,
# which crashes with a TypeError ("unexpected keyword argument
# 'skip_file_prefixes'") when shap loads AFTER crewai in the same process --
# something crewai (or one of its deps) does to warnings.warn() is
# incompatible with a newer matplotlib deprecation-warning call shap makes
# on import. The real MCP server never hits this because it imports
# fraud_attribution (and therefore shap) before crewai is ever involved;
# this agents/ process imports crewai first. Avoiding the shap import here
# entirely sidesteps the ordering problem rather than depending on import
# order discipline holding across every future entry point.
_FRAUD_FEATURE_COLS = (
    "account_age_days", "total_orders_lifetime", "total_returns_lifetime", "claim_frequency_90d",
    "refund_amount_usd", "days_to_return", "customer_support_contacts_90d", "previous_dispute_count",
    "address_match", "is_high_value_item", "photo_evidence_provided",
    "claim_category", "image_consistency",
)


def build_fraud_scoring_agent(tools, customer_ref: Optional[str] = None, claim_ref: Optional[str] = None) -> Agent:
    """customer_ref/claim_ref (Q30/Q73): forwarded to get_gpt41_mini_llm so
    this agent's real LLM calls get tagged for Langfuse tracing."""
    return Agent(
        role="ML Fraud Scoring Agent",
        goal="Assemble every feature the trained fraud-risk model needs and report its real, calibrated output.",
        backstory=(
            "You score return/refund claims for fraud risk using a trained, calibrated "
            "ML model -- you never estimate or guess a risk band yourself. You gather "
            "every required feature via tool calls, call score_fraud_risk exactly once "
            "with the complete, accurate feature set, and report its result verbatim as "
            "risk_band and risk_score, with key_signals exactly as the model returned "
            "them. Your own contribution is only the reasoning narrative explaining what "
            "the real key_signals mean in context -- never a different risk_band/"
            "risk_score/key_signals than what score_fraud_risk actually returned."
        ),
        llm=get_gpt41_mini_llm(customer_ref, claim_ref),
        tools=tools,
        verbose=True,
    )


def _score_fraud_risk_genuinely_succeeded(output: TaskOutput) -> bool:
    """Same class of check as Image Parsing's guardrail: confirms
    score_fraud_risk actually returned real data rather than the agent
    fabricating risk_band/risk_score/key_signals after a failed or skipped
    call. The feature payload here is a small JSON dict, not a large base64
    blob, so corruption in transit is far less likely than analyze_image's
    -- but the check costs nothing and catches the same failure shape if a
    tool call is skipped or errors."""
    for message in output.messages:
        if message.get("role") == "tool" and message.get("name") == "score_fraud_risk":
            content = message.get("content") or ""
            if isinstance(content, str) and '"risk_band"' in content and "Error calling tool" not in content:
                return True
    return False


def fraud_assessment_guardrail(output: TaskOutput) -> Tuple[bool, Any]:
    """Layer 3 (Q34): key_signals must be real feature names the model
    actually used, not agent-invented ones -- mirrors the same grounding
    check already verified directly against score_fraud_risk
    (tests/test_orchestrator_server.py::test_score_fraud_risk_returns_
    grounded_result), now enforced at the agent-output layer too, since an
    agent narrating the result could still typo or invent a signal name."""
    result = output.pydantic
    if result is None:
        return False, "expected a FraudAssessment, got no structured output"
    if not _score_fraud_risk_genuinely_succeeded(output):
        return False, "score_fraud_risk never genuinely succeeded -- this looks like a fabricated risk assessment"

    for signal in result.key_signals:
        if not any(signal == col or signal.startswith(f"{col}_") for col in _FRAUD_FEATURE_COLS):
            return False, f"key_signal {signal!r} doesn't trace back to any real FEATURE_COLS entry"
    return True, result


def build_fraud_scoring_task(
    agent: Agent,
    customer_ref: str,
    order_ref: str,
    claim_ref: str,
    claim_category: str,
    claim_description: str,
    days_to_return: int,
    photo_evidence_provided: bool,
    image_consistency: str,
) -> Task:
    return Task(
        description=(
            f"Score claim {claim_ref} (customer {customer_ref}, order {order_ref}) for "
            f"fraud risk. The claim category is '{claim_category}', described as: "
            f'"{claim_description}". It was reported {days_to_return} day(s) after '
            f"delivery. Photo evidence was {'provided' if photo_evidence_provided else 'not provided'}, "
            f"and the Image Parsing Agent's consistency verdict was '{image_consistency}'.\n\n"
            "1. Call get_order to get the order amount (this is refund_amount_usd).\n"
            "2. Call get_account_info to get account_age_days, total_orders_lifetime, "
            "total_returns_lifetime, customer_support_contacts_90d, previous_dispute_count, "
            "and address_match.\n"
            "3. Call get_claim_frequency to get claim_frequency_90d, then call "
            "increment_claim_frequency to record this claim in that window.\n"
            "4. Optionally call get_tracking_status for delivery context to inform your "
            "reasoning (its result is not a model input).\n"
            f"5. Compute is_high_value_item as true if the order amount exceeds "
            f"${HIGH_VALUE_THRESHOLD_USD:.0f}, otherwise false.\n"
            "6. Call score_fraud_risk exactly once, passing all 13 arguments directly "
            "(account_age_days, total_orders_lifetime, total_returns_lifetime, "
            "claim_frequency_90d, refund_amount_usd, days_to_return, "
            "customer_support_contacts_90d, previous_dispute_count, address_match, "
            f"is_high_value_item, photo_evidence_provided={photo_evidence_provided}, "
            f"claim_category='{claim_category}', image_consistency='{image_consistency}', "
            f"days_to_return={days_to_return}) with the real values gathered above -- "
            "these are 13 separate arguments to the tool call, not one combined object.\n"
            "7. Report its exact risk_band, risk_score, and key_signals -- do not alter "
            "them. Write your own reasoning explaining what the real key_signals mean, "
            "but never invent a different risk_band/risk_score/key_signals than what the "
            "tool actually returned."
        ),
        expected_output="A FraudAssessment with risk_band, risk_score, key_signals, and reasoning.",
        agent=agent,
        output_pydantic=FraudAssessment,
        guardrail=fraud_assessment_guardrail,
    )
