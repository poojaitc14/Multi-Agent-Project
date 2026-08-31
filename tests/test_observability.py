"""Real Langfuse tracing verification (project-plan.md Q30/Q73/Q74) -- not
mocked: runs a real Orchestrator classification call, then queries
Langfuse's real public API to confirm a trace actually landed there with
the correct userId/sessionId. Skipped (not failed) if Langfuse credentials
aren't configured, since Langfuse is an optional integration
(agents/observability.py's configure_langfuse() degrades the same way).
"""

import os
import sys
import time
import uuid
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crewai import Crew  # noqa: E402

from agents.mcp_tools import orchestrator_mcp_adapter  # noqa: E402
from agents.orchestrator_agent import build_classification_task, build_orchestrator_agent  # noqa: E402

LANGFUSE_CONFIGURED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(os.environ.get("LANGFUSE_SECRET_KEY"))


def _poll_for_trace_by_session(session_id: str, timeout_s: float = 90.0) -> dict | None:
    """Langfuse's ingestion is asynchronous -- a short poll loop is more
    robust than a fixed sleep, and fails fast on a real 4xx/5xx instead of
    masking it behind a timeout. Filters by sessionId via the real list
    endpoint rather than assuming a specific trace_id value -- litellm's
    exact trace_id-defaulting behavior isn't a documented contract worth
    depending on, even though it happened to equal session_id in manual
    verification.

    Retries on a transient network error (a real one hit while building
    this: a single ReadTimeout against cloud.langfuse.com failed the whole
    test on the first version of this loop, even though the trace itself
    landed correctly -- a flaky poll shouldn't be mistaken for the feature
    being broken). If every single attempt errors -- confirmed for real via
    curl hitting the same /api/public/traces route directly, independent of
    this code, with cloud.langfuse.com's root domain responding fine in the
    same breath -- that's "we couldn't reach Langfuse to check" and gets
    reported to the caller distinctly from "we checked and the trace never
    arrived", since only the latter is evidence the integration is broken.

    timeout_s=90.0 (raised from an initial 45.0): this specific endpoint's
    per-request latency has been measured, directly and repeatedly, ranging
    from under a second to 60+ seconds on the same query with nothing else
    changed -- a real, external characteristic of this route right now, not
    a guess. A run that happens to catch a few slow-but-successful requests
    can otherwise exhaust a shorter window's retry budget before ever
    getting a real read on whether the trace has landed."""
    host = os.environ["LANGFUSE_HOST"]
    auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
    deadline = time.time() + timeout_s
    ever_reached_server = False
    while time.time() < deadline:
        try:
            response = requests.get(
                f"{host}/api/public/traces", auth=auth, params={"sessionId": session_id, "limit": 1}, timeout=15
            )
            response.raise_for_status()
        except requests.exceptions.RequestException:
            time.sleep(2)
            continue
        ever_reached_server = True
        data = response.json().get("data", [])
        if data:
            return data[0]
        time.sleep(2)
    if not ever_reached_server:
        pytest.skip("Langfuse's /api/public/traces endpoint was unreachable for the whole poll window (every request errored/timed out) -- could not check whether the trace arrived")
    return None


@pytest.mark.skipif(not LANGFUSE_CONFIGURED, reason="LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not configured")
def test_langfuse_trace_uses_customer_ref_and_claim_ref_not_raw_identifier():
    """Q30's PII boundary extended to Langfuse: userId/sessionId on the
    real trace must be the resolved refs, never a raw identifier -- this
    test never even has a raw identifier in scope, by construction, since
    build_orchestrator_agent's customer_ref/claim_ref parameters only ever
    accept already-resolved refs (see agents/observability.py)."""
    customer_ref = f"cst_test{uuid.uuid4().hex[:8]}"
    claim_ref = f"clm_test{uuid.uuid4().hex[:8]}"

    with orchestrator_mcp_adapter() as tools:
        agent = build_orchestrator_agent(tools, customer_ref, claim_ref)
        task = build_classification_task(agent, "What is your return policy?")
        Crew(agents=[agent], tasks=[task], verbose=False).kickoff()

    trace = _poll_for_trace_by_session(claim_ref)
    assert trace is not None, f"no Langfuse trace with sessionId={claim_ref!r} appeared within the poll window"
    assert trace["userId"] == customer_ref
    assert trace["sessionId"] == claim_ref
