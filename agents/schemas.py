"""Structured output schemas for the 4 agents (project-plan.md's "Structured
agent outputs" section). Every CrewAI Task uses one of these as
output_pydantic -- Layer 2 of the 3-layer guardrail design (Layer 1 is
constrained decoding at the model itself; Layer 3 is a guardrail function
per Task checking business rules a schema alone can't express)."""

from typing import Literal, Optional

from pydantic import BaseModel

# project-plan.md's "Claim categories" (Refund policy section) -- single
# source of truth, referenced by ClaimIntakeResult below, backend/main.py's
# request schema, and agents/flow.py's PHOTO_REQUIRED_CATEGORIES, so the
# five real values are never duplicated/allowed to drift.
ClaimCategory = Literal[
    "Damaged in Transit", "Wrong Item Received", "Not as Described", "Defective/DOA", "Change of Mind",
]


class ClassificationResult(BaseModel):
    request_type: Literal["new_claim", "follow_up", "general_inquiry"]
    order_ref: Optional[str] = None


class ClaimIntakeResult(BaseModel):
    """Slice 11: extracts whatever claim details a customer message
    contains, and asks a follow_up_question for whatever's still missing --
    multi-turn, not single-shot (see agents/orchestrator_agent.py's
    build_intake_task). All fields besides request_type are optional
    because a first message often won't have everything."""

    request_type: Literal["new_claim", "follow_up", "general_inquiry"]
    order_ref: Optional[str] = None
    claim_category: Optional[ClaimCategory] = None
    claim_description: Optional[str] = None
    days_to_return: Optional[int] = None
    follow_up_question: Optional[str] = None


class ConsistencyAssessment(BaseModel):
    verdict: Literal["consistent", "partially_consistent", "inconsistent", "no_photo"]
    product_match: bool
    reasoning: str


class FraudAssessment(BaseModel):
    risk_band: Literal["low", "medium", "high"]
    risk_score: float
    key_signals: list[str]
    reasoning: str


class Verdict(BaseModel):
    decision: Literal["approve", "deny", "escalate"]
    refund_amount: Optional[float] = None
    refund_form: Optional[Literal["original_payment_method", "store_credit"]] = None
    policy_clause: Optional[str] = None
    policy_version: Optional[str] = None
    reasoning: str
