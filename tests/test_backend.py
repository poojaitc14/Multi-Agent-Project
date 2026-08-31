"""Integration tests for backend/main.py -- real MCP server, real Azure
OpenAI, no mocking, same philosophy as tests/test_orchestrator_server.py.
Requires mcp-servers/orchestrator_server.py already running (real process,
real port), just like the agents/ layer it wraps.
"""

import os
import sys
import uuid
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.mcp_tools import orchestrator_mcp_adapter  # noqa: E402
from backend.main import app  # noqa: E402

load_dotenv()

client = TestClient(app)
REVIEWER_PASSWORD = os.environ["REVIEWER_PASSWORD"]
REVIEWER_HEADERS = {"X-Reviewer-Password": REVIEWER_PASSWORD}


def _seed_review_queue_row(claim_ref: str, customer_ref: str) -> None:
    """Q72's insert-on-escalate path is triggered by a real Decision Agent
    verdict, which isn't reliably reproducible yet (Q70) -- these tests
    verify the reviewer-facing endpoints for real (list/approve/deny/
    issue_refund/idempotency) against a directly-seeded row instead,
    exactly as this was first verified by hand while building it."""
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO review_queue (claim_ref, order_ref, customer_ref, claim_category, "
                "claim_description, refund_amount_usd, image_verdict, fraud_risk_band, verdict_reasoning) "
                "VALUES (%s, '1', %s, 'Damaged in Transit', 'test row', 25.00, 'consistent', 'high', 'test')",
                (claim_ref, customer_ref),
            )
        conn.commit()


def _resolve_test_customer_ref() -> str:
    with orchestrator_mcp_adapter() as tools:
        resolve = next(t for t in tools if t.name == "resolve_customer_ref")
        return resolve.run(customer_id_or_email=f"reviewqueue-test-{uuid.uuid4().hex[:8]}@example.com")


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_submit_change_of_mind_claim_runs_real_pipeline():
    """Change of Mind skips Image Parsing (Q65/Q66) and runs Fraud Scoring
    for real -- confirms the whole HTTP -> Flow -> real-agents path works,
    accepting any real outcome (including 'decision_unavailable', since a
    genuine API/guardrail failure degrades to that rather than a 500)."""
    response = client.post(
        "/claims",
        json={
            "customer_identifier": f"backend-test-{uuid.uuid4().hex[:8]}@example.com",
            "order_ref": "1",
            "claim_category": "Change of Mind",
            "claim_description": "No longer wants the item, unopened.",
            "days_to_return": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["claim_ref"].startswith("clm_")
    assert data["photo_required"] is False
    assert data["image_verdict"] == "consistent"
    assert data["fraud_risk_band"] in ("low", "medium", "high")
    assert data["outcome"] in ("decision_unavailable", "approved", "deny", "escalate")
    # Q27: raw customer identifier must never appear in the response
    assert "customer_identifier" not in data
    assert "customer_ref" not in data


def test_submit_photo_required_claim_with_no_photo_reprompts():
    response = client.post(
        "/claims",
        json={
            "customer_identifier": f"backend-test2-{uuid.uuid4().hex[:8]}@example.com",
            "order_ref": "1",
            "claim_category": "Damaged in Transit",
            "claim_description": "Item arrived crushed.",
            "days_to_return": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["photo_required"] is True
    assert data["image_verdict"] == "no_photo"
    assert data["outcome"] == "re_prompt_for_photo"
    assert data["fraud_risk_band"] is None, "Fraud Scoring must never run on a re-prompt"


def test_invalid_claim_category_rejected():
    response = client.post(
        "/claims",
        json={
            "customer_identifier": "badcat@example.com",
            "order_ref": "1",
            "claim_category": "Not A Real Category",
            "claim_description": "test",
            "days_to_return": 1,
        },
    )
    assert response.status_code == 422


def test_rate_limit_blocks_fourth_submission_within_an_hour():
    """Q39/Q45: 3 per customer_ref per hour. Uses Change of Mind (cheapest
    real path) 4 times for the same customer_identifier."""
    email = f"ratelimit-test-{uuid.uuid4().hex[:8]}@example.com"
    payload = {
        "customer_identifier": email,
        "order_ref": "1",
        "claim_category": "Change of Mind",
        "claim_description": "test",
        "days_to_return": 1,
    }
    statuses = [client.post("/claims", json=payload).status_code for _ in range(4)]
    assert statuses[:3] == [200, 200, 200]
    assert statuses[3] == 429


def test_messages_multi_turn_intake_completes_claim():
    """Slice 11: a first message with a missing field gets a real
    follow_up_question and needs_more_info=true; the answer, sent as a
    second message, is merged with what get_conversation_state persisted
    from the first and completes the same claim -- real multi-turn, not
    single-shot, using the real conversation-state MCP tools."""
    email = f"messages-test-{uuid.uuid4().hex[:8]}@example.com"

    first = client.post(
        "/messages",
        json={"customer_identifier": email, "message": "Hi, my order 1 arrived damaged, the box was crushed."},
    )
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["needs_more_info"] is True
    assert first_data["request_type"] == "new_claim"
    assert first_data["follow_up_question"]
    assert first_data["claim_result"] is None

    second = client.post("/messages", json={"customer_identifier": email, "message": "It arrived 5 days ago."})
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["needs_more_info"] is False
    assert second_data["claim_result"] is not None
    assert second_data["claim_result"]["claim_ref"].startswith("clm_")
    assert second_data["claim_result"]["photo_required"] is True
    # Q27: raw customer identifier must never appear in the response
    assert "customer_identifier" not in second_data
    assert "customer_ref" not in second_data


def test_messages_follow_ups_within_one_intake_do_not_count_against_rate_limit():
    """Q39, revised for multi-turn intake: only a genuinely new claim
    counts against the 3/hour cap. 3 messages that are all part of ONE
    continued intake, followed by a real new claim, must all succeed --
    if the exemption didn't work, the 4th (a real new claim) would 429."""
    email = f"messages-ratelimit-{uuid.uuid4().hex[:8]}@example.com"

    r1 = client.post("/messages", json={"customer_identifier": email, "message": "I want to return something."})
    assert r1.status_code == 200
    assert r1.json()["needs_more_info"] is True

    r2 = client.post(
        "/messages", json={"customer_identifier": email, "message": "Order 1, its damaged in transit."}
    )
    assert r2.status_code == 200
    assert r2.json()["needs_more_info"] is True

    r3 = client.post(
        "/messages", json={"customer_identifier": email, "message": "It was 4 days ago. The box was crushed."}
    )
    assert r3.status_code == 200
    assert r3.json()["needs_more_info"] is False

    r4 = client.post(
        "/messages",
        json={"customer_identifier": email, "message": "I have another order too, order 1, change of mind, 2 days ago."},
    )
    assert r4.status_code == 200, "a genuinely new claim must still succeed -- the 3 prior messages were one continued intake, not 3 new claims"


def test_messages_genuinely_new_claims_still_rate_limited():
    """The exemption above must not make the limit toothless: 3
    independently-complete new claims (each resolved in one message) use
    up the real budget, and a 4th genuinely new claim gets 429."""
    email = f"messages-ratelimit2-{uuid.uuid4().hex[:8]}@example.com"
    message = "Order 1, change of mind, 2 days ago, dont want it."

    statuses = [
        client.post("/messages", json={"customer_identifier": email, "message": message}).status_code
        for _ in range(4)
    ]
    assert statuses[:3] == [200, 200, 200]
    assert statuses[3] == 429


def test_tool_registry_requires_reviewer_password():
    unauthenticated = client.get("/tool-registry")
    assert unauthenticated.status_code == 401

    authenticated = client.get("/tool-registry", headers=REVIEWER_HEADERS)
    assert authenticated.status_code == 200
    registry = authenticated.json()
    assert set(registry.keys()) == {"Orchestrator", "Image Parsing Agent", "Fraud Scoring Agent", "Decision Agent"}
    assert "apply_decision_matrix" in registry["Decision Agent"]
    assert "issue_refund" in registry["Orchestrator"]


def test_review_queue_requires_reviewer_password():
    unauthenticated = client.get("/review-queue")
    assert unauthenticated.status_code == 401


def test_review_queue_approve_issues_real_refund():
    """Q72: a reviewer's approval must flow through the same gated
    issue_refund MCP call the Orchestrator itself uses -- verified by
    checking the real refund_transactions row it writes, not just trusting
    the API's own response."""
    claim_ref = f"clm_test_{uuid.uuid4().hex[:8]}"
    customer_ref = _resolve_test_customer_ref()
    _seed_review_queue_row(claim_ref, customer_ref)

    listing = client.get("/review-queue", headers=REVIEWER_HEADERS)
    assert listing.status_code == 200
    assert any(item["claim_ref"] == claim_ref for item in listing.json())

    decision = client.post(f"/review-queue/{claim_ref}/decide", headers=REVIEWER_HEADERS, json={"approve": True})
    assert decision.status_code == 200
    data = decision.json()
    assert data["status"] == "approved"
    assert data["transaction_id"]

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT amount, transaction_id FROM refund_transactions WHERE claim_ref = %s", (claim_ref,)
            )
            row = cur.fetchone()
    assert row is not None, "approval must write a real refund_transactions row"
    assert row[1] == data["transaction_id"]

    listing_after = client.get("/review-queue", headers=REVIEWER_HEADERS)
    assert not any(item["claim_ref"] == claim_ref for item in listing_after.json())

    repeat = client.post(f"/review-queue/{claim_ref}/decide", headers=REVIEWER_HEADERS, json={"approve": True})
    assert repeat.status_code == 409


def test_review_queue_deny_issues_no_refund():
    claim_ref = f"clm_test_{uuid.uuid4().hex[:8]}"
    customer_ref = _resolve_test_customer_ref()
    _seed_review_queue_row(claim_ref, customer_ref)

    decision = client.post(f"/review-queue/{claim_ref}/decide", headers=REVIEWER_HEADERS, json={"approve": False})
    assert decision.status_code == 200
    data = decision.json()
    assert data["status"] == "denied"
    assert data["transaction_id"] is None

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM refund_transactions WHERE claim_ref = %s", (claim_ref,))
            count = cur.fetchone()[0]
    assert count == 0, "a denial must never write a refund_transactions row"


def test_review_queue_decide_unknown_claim_404s():
    response = client.post(
        "/review-queue/clm_does_not_exist/decide", headers=REVIEWER_HEADERS, json={"approve": True}
    )
    assert response.status_code == 404
