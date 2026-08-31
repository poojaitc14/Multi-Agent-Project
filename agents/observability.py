"""Langfuse tracing (project-plan.md Q30/Q73). Every agent's LLM call
routes through litellm already (agents/llms.py's is_litellm=True), so
litellm's own Langfuse callback is the integration point -- no separate
Langfuse SDK instrumentation needed for the calls this covers.

Q30's PII boundary, extended to Langfuse specifically: every trace's
user_id/session_id must be customer_ref/claim_ref, never a raw customer
identifier -- litellm's real (source-verified, not assumed) metadata
contract for this is {"trace_user_id": ..., "session_id": ...}, passed via
crewai.LLM's additional_params (which litellm.py's own _completion_params
unpacks with **self.additional_params directly into the litellm.completion()
call -- confirmed against crewai's real source, not assumed from docs,
after Q52's LLM(endpoint=...) mistake taught not to trust doc pages here).

Real, honest coverage gap, not silently glossed over: mcp-servers/
orchestrator_server.py's analyze_image calls Azure OpenAI directly via the
raw AzureOpenAI SDK client, not litellm -- so that one call (the actual
vision judgment, which is the only place a real photo's bytes ever reach
an LLM) is NOT captured by this Langfuse integration at all. This
incidentally satisfies Q30's "no unredacted photo in any trace" rule by
non-coverage rather than by a redaction step -- worth knowing, not
something to "fix" by adding photo bytes to a trace.
"""

import os

import litellm

_configured = False


def configure_langfuse() -> bool:
    """Call once, at process startup (mirrors FraudAttributor's eager-
    warmup pattern in orchestrator_server.py -- a one-time setup cost, not
    per-call). Returns False (and does nothing) if Langfuse credentials
    aren't configured, so this is always safe to call even when Langfuse
    hasn't been set up yet -- real, honest degradation, not a crash."""
    global _configured
    if _configured:
        return True
    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        return False
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]
    _configured = True
    return True


def trace_metadata(customer_ref: str, claim_ref: str | None = None) -> dict:
    """additional_params= for crewai.LLM -- customer_ref becomes the
    Langfuse trace's user_id, claim_ref (when known -- not yet assigned
    during multi-turn intake, before a claim exists) becomes its
    session_id. Never a raw customer identifier (Q27/Q30): callers must
    only ever pass a resolved customer_ref here, never the raw email/ID."""
    metadata: dict = {"trace_user_id": customer_ref}
    if claim_ref:
        metadata["session_id"] = claim_ref
    return {"metadata": metadata}
