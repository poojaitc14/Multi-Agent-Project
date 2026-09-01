"""Integration tests for the Orchestrator MCP server (Slice 1 / now the
single MCP server, Q55).

Not mocked, per the plan: these call the real server (in-memory FastMCP
transport for most tools; real HTTP for issue_refund specifically, since
its Q55 bearer-token gate is an HTTP-layer concern -- see "Real HTTP tests"
below), real local Postgres, real AWS DynamoDB/S3 (dev-prefixed resources,
project-plan.md Q51 -- LocalStack was dropped), and the real DummyJSON
public API. Requires `docker compose -f infra/docker-compose.yml up -d
postgres` to be running first, plus real AWS credentials configured (env
vars, ~/.aws/credentials, or an IAM role) with access to the dev-prefixed
table/bucket names in .env.
"""

import base64
import os
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastmcp import Client

from conftest import http_client as _http_client  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp-servers"))
import orchestrator_server as srv  # noqa: E402


@pytest.fixture
async def client():
    async with Client(transport=srv.mcp) as c:
        yield c


async def test_resolve_customer_ref_creates_and_reuses(client: Client):
    identifier = f"test-{uuid.uuid4().hex[:8]}@example.com"

    first = await client.call_tool("resolve_customer_ref", {"customer_id_or_email": identifier})
    second = await client.call_tool("resolve_customer_ref", {"customer_id_or_email": identifier})

    assert first.data == second.data
    assert first.data.startswith("cst_")


async def test_resolve_customer_ref_different_identifiers_get_different_refs(client: Client):
    a = await client.call_tool(
        "resolve_customer_ref", {"customer_id_or_email": f"a-{uuid.uuid4().hex[:8]}@example.com"}
    )
    b = await client.call_tool(
        "resolve_customer_ref", {"customer_id_or_email": f"b-{uuid.uuid4().hex[:8]}@example.com"}
    )
    assert a.data != b.data


async def test_get_order_returns_deidentified_shape(client: Client):
    result = await client.call_tool("get_order", {"order_ref": "1"})
    order = result.data

    assert "product" in order
    assert "amount" in order
    assert order["amount"] > 0
    # PII boundary (Q27): no raw shipping/payment/customer fields
    for forbidden in ("name", "email", "address", "payment", "phone"):
        assert forbidden not in order


async def test_conversation_state_roundtrip_with_ttl(client: Client):
    customer_ref = f"cst_{uuid.uuid4().hex[:10]}"
    await client.call_tool(
        "put_conversation_state",
        {"customer_ref": customer_ref, "state": {"claim_ref": "clm_test", "claim_status": "awaiting_photo"}},
    )

    result = await client.call_tool("get_conversation_state", {"customer_ref": customer_ref})
    item = result.data

    assert item["claim_status"] == "awaiting_photo"
    assert "ttl" in item  # DynamoDB TTL attribute (Q44: 14 days)


async def test_photo_roundtrip(client: Client):
    claim_ref = f"clm_{uuid.uuid4().hex[:8]}"
    original = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
    original_b64 = base64.b64encode(original).decode("ascii")

    await client.call_tool("store_photo", {"claim_ref": claim_ref, "photo_base64": original_b64})
    result = await client.call_tool("get_photo", {"claim_ref": claim_ref})

    assert base64.b64decode(result.data) == original


async def test_transcript_store_does_not_raise(client: Client):
    await client.call_tool(
        "store_transcript",
        {"customer_ref": f"cst_{uuid.uuid4().hex[:10]}", "transcript": "customer: hi\nagent: hello"},
    )


async def test_issue_refund_is_idempotent():
    """Real HTTP, Orchestrator's own token -- the realistic calling path
    now that Q55's gate is in place."""
    order_ref = f"ord_{uuid.uuid4().hex[:8]}"
    claim_ref = f"clm_{uuid.uuid4().hex[:8]}"

    async with _http_client(srv.ORCHESTRATOR_MCP_TOKEN) as c:
        first = await c.call_tool(
            "issue_refund",
            {"order_ref": order_ref, "claim_ref": claim_ref, "amount": 42.50, "reason": "test refund"},
        )
        second = await c.call_tool(
            "issue_refund",
            {"order_ref": order_ref, "claim_ref": claim_ref, "amount": 42.50, "reason": "test refund"},
        )

    assert first.data["status"] == "refunded"
    assert second.data["status"] == "already_refunded"
    assert first.data["transaction_id"] == second.data["transaction_id"]

    with srv._pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM refund_transactions WHERE order_ref = %s AND claim_ref = %s",
                (order_ref, claim_ref),
            )
            assert cur.fetchone()[0] == 1


async def test_issue_refund_rejects_wrong_agent_token():
    """Q55: another agent's real, valid-for-other-tools token must still
    fail specifically for issue_refund -- this is the actual security
    property the whole gate exists for."""
    other_agent_token = os.environ["IMAGE_PARSING_MCP_TOKEN"]
    order_ref = f"ord_{uuid.uuid4().hex[:8]}"
    claim_ref = f"clm_{uuid.uuid4().hex[:8]}"

    async with _http_client(other_agent_token) as c:
        with pytest.raises(Exception, match="restricted to the Orchestrator"):
            await c.call_tool(
                "issue_refund",
                {"order_ref": order_ref, "claim_ref": claim_ref, "amount": 42.50, "reason": "test refund"},
            )

    with srv._pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM refund_transactions WHERE order_ref = %s AND claim_ref = %s",
                (order_ref, claim_ref),
            )
            assert cur.fetchone()[0] == 0, "rejected call must not have written a refund row"


async def test_issue_refund_rejects_missing_token():
    order_ref = f"ord_{uuid.uuid4().hex[:8]}"
    claim_ref = f"clm_{uuid.uuid4().hex[:8]}"

    async with _http_client(None) as c:
        with pytest.raises(Exception, match="restricted to the Orchestrator"):
            await c.call_tool(
                "issue_refund",
                {"order_ref": order_ref, "claim_ref": claim_ref, "amount": 42.50, "reason": "test refund"},
            )


async def test_other_tools_unaffected_by_the_gate(client: Client):
    """The gate only checks issue_refund (per its docstring) -- confirm a
    completely unauthenticated in-memory call still works for a tool that
    was never meant to be gated."""
    result = await client.call_tool("get_order", {"order_ref": "1"})
    assert "product" in result.data


async def test_search_refund_policy_registered_and_ungated(client: Client):
    """Q56: search_refund_policy is real (ml/rag/), not mocked -- but the
    OpenSearch Serverless collection it depends on gets torn down between
    sessions (real, billable AWS infra), so this handles both states rather
    than assuming one. Either way, confirms it's registered, callable with
    no token at all (Q56: not gated, unlike issue_refund), and doesn't hang."""
    try:
        result = await client.call_tool(
            "search_refund_policy", {"query": "What happens if my refund is over $200?"}
        )
        data = result.data
        assert "confident" in data
        if data["confident"]:
            assert len(data["chunks"]) > 0
            assert any(c["chunk_id"] == "guardrail" for c in data["chunks"])
        else:
            assert data["action"] == "escalate"
    except Exception as e:
        assert "doesn't exist" in str(e), f"expected the collection-missing error, got: {e!r}"


async def test_score_fraud_risk_returns_grounded_result(client: Client):
    """Q7/Q29/Q34/Q58: real registered model, real SHAP attribution, no
    mocking. risk_band is a direct class label (not a thresholded score,
    Q58); key_signals are real feature names the model actually used.

    13 flat parameters, not a features: dict (Slice 7) -- a generic dict
    parameter proved empirically unreliable through real CrewAI tool-calling
    (the agent sent an empty dict on 15/15 real attempts)."""
    sample = {
        "account_age_days": 30, "total_orders_lifetime": 3, "total_returns_lifetime": 2,
        "claim_frequency_90d": 1, "refund_amount_usd": 89.99, "days_to_return": 5,
        "customer_support_contacts_90d": 0, "previous_dispute_count": 0,
        "address_match": True, "is_high_value_item": False, "photo_evidence_provided": True,
        "claim_category": "Damaged in Transit", "image_consistency": "consistent",
    }
    result = await client.call_tool("score_fraud_risk", sample)
    data = result.data

    assert data["risk_band"] in ("low", "medium", "high")
    assert 0.0 <= data["risk_score"] <= 1.0
    assert len(data["key_signals"]) == 3
    # key_signals must be real feature names, not anything the model wasn't
    # actually trained on -- the whole point of Q34's grounding requirement
    from fraud_attribution import FEATURE_COLS  # noqa: PLC0415

    for signal in data["key_signals"]:
        assert any(signal == col or signal.startswith(f"{col}_") for col in FEATURE_COLS), (
            f"key_signal {signal!r} doesn't trace back to any real FEATURE_COLS entry"
        )


async def test_get_account_info_seeds_once_and_is_stable(client: Client):
    """Fraud Scoring's account-level features have no real data source in
    this project (DummyJSON has no per-customer lifetime history) -- the
    first call seeds a synthetic profile, and every later call for the same
    customer_ref must return that same seeded profile, not a fresh sample."""
    customer_ref = await client.call_tool(
        "resolve_customer_ref", {"customer_id_or_email": f"acct-{uuid.uuid4().hex[:8]}@example.com"}
    )
    customer_ref = customer_ref.data

    first = await client.call_tool("get_account_info", {"customer_ref": customer_ref})
    second = await client.call_tool("get_account_info", {"customer_ref": customer_ref})

    assert first.data == second.data
    for key in (
        "account_age_days", "total_orders_lifetime", "total_returns_lifetime",
        "customer_support_contacts_90d", "previous_dispute_count", "address_match",
    ):
        assert key in first.data
    assert isinstance(first.data["address_match"], bool)


async def test_get_account_info_rejects_unknown_customer_ref(client: Client):
    with pytest.raises(Exception, match="unknown customer_ref"):
        await client.call_tool("get_account_info", {"customer_ref": "cst_doesnotexist"})


async def test_claim_frequency_roundtrip(client: Client):
    customer_ref = f"cst_{uuid.uuid4().hex[:10]}"
    claim_ref = f"clm_{uuid.uuid4().hex[:8]}"

    before = await client.call_tool("get_claim_frequency", {"customer_ref": customer_ref})
    await client.call_tool("increment_claim_frequency", {"customer_ref": customer_ref, "claim_ref": claim_ref})
    after = await client.call_tool("get_claim_frequency", {"customer_ref": customer_ref})

    assert before.data["claim_frequency_90d"] == 0
    assert after.data["claim_frequency_90d"] == 1


async def test_get_tracking_status_degrades_gracefully(client: Client):
    """order_ref is a DummyJSON cart ID, not a real carrier tracking number
    -- delivery_status: 'unknown' is the expected, correct result here, not
    a failure to work around."""
    result = await client.call_tool("get_tracking_status", {"order_ref": "1"})
    assert result.data["delivery_status"] == "unknown"
    assert "address_match" not in result.data  # moved to get_account_info


async def test_get_product_reference_real_dummyjson_data(client: Client):
    result = await client.call_tool("get_product_reference", {"order_ref": "1"})
    data = result.data
    assert data["title"] and data["title"] != "unknown"
    assert data["description"]
    assert data["reference_image_url"].startswith("https://")


async def test_redact_photo_removes_detectable_text(client: Client):
    """Real OpenCV + EasyOCR, not mocked -- generates a photo with a
    tracking-number-like text overlay, confirms EasyOCR can read it in the
    original, then confirms nothing is readable after redact_photo runs."""
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 200), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "TRACKING NUMBER LABEL TEXT", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    original_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    original_img = cv2.imdecode(np.frombuffer(base64.b64decode(original_b64), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert srv._ocr_reader.readtext(original_img), "sanity check: OCR must detect text in the un-redacted original"

    result = await client.call_tool("redact_photo", {"photo_base64": original_b64})
    redacted_b64 = result.data

    redacted_img = cv2.imdecode(np.frombuffer(base64.b64decode(redacted_b64), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert not srv._ocr_reader.readtext(redacted_img), "text is still readable after redact_photo"


async def test_redact_photo_rejects_invalid_base64(client: Client):
    with pytest.raises(Exception, match="not valid base64"):
        await client.call_tool("redact_photo", {"photo_base64": "not-valid-base64!!!"})


async def test_analyze_image_returns_grounded_verdict(client: Client):
    """Real Azure OpenAI GPT-4.1 mini vision call, not mocked. Uses a photo
    that obviously does not match the claimed product, so the verdict
    should clearly be inconsistent/no product match -- a weak assertion
    (just checking the shape) wouldn't actually confirm the vision call is
    grounded in the real image content."""
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (300, 300), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 250, 250], fill=(200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    photo_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    result = await client.call_tool(
        "analyze_image",
        {
            "photo_base64": photo_b64,
            "claim_category": "Wrong Item Received",
            "claim_description": "I ordered a laptop but received this instead.",
            "reference_title": "15-inch Gaming Laptop",
            "reference_description": "A black gaming laptop with a 15-inch display.",
        },
    )
    data = result.data
    assert data["verdict"] in ("consistent", "partially_consistent", "inconsistent")
    assert isinstance(data["product_match"], bool)
    assert isinstance(data["reasoning"], str) and len(data["reasoning"]) > 0
    assert data["product_match"] is False, "a plain red square should not be judged to match a laptop"


async def test_analyze_claim_photo_no_photo_is_deterministic(client: Client):
    """project-plan.md Q86/Q87: analyze_claim_photo collapses get_photo ->
    redact_photo -> get_product_reference -> analyze_image into one
    server-side tool. No photo stored for this claim_ref -> a real S3
    NoSuchKey, handled deterministically (no LLM call, no vision API cost)."""
    claim_ref = f"clm_{uuid.uuid4().hex[:8]}"
    result = await client.call_tool(
        "analyze_claim_photo",
        {
            "claim_ref": claim_ref, "order_ref": "1", "claim_category": "Damaged in Transit",
            "claim_description": "The box arrived crushed.",
        },
    )
    data = result.data
    assert data == {
        "verdict": "no_photo", "product_match": False,
        "reasoning": f"No photo was found for claim {claim_ref}.",
    }


async def test_analyze_claim_photo_real_photo_returns_grounded_verdict(client: Client):
    """The real, end-to-end fix for the base64-truncation bug: a real photo
    stored via store_photo, then analyze_claim_photo does the entire fetch/
    redact/analyze chain itself -- the caller here (this test, standing in
    for the Image Parsing Agent) never handles the raw photo bytes at all,
    only claim_ref. Same grounding assertion as
    test_analyze_image_returns_grounded_verdict: a claim photo that
    obviously doesn't match the referenced product should score
    product_match=False, confirming analyze_claim_photo's internal
    analyze_image call is genuinely grounded in the real image content."""
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (300, 300), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 250, 250], fill=(200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    photo_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    claim_ref = f"clm_{uuid.uuid4().hex[:8]}"
    await client.call_tool("store_photo", {"claim_ref": claim_ref, "photo_base64": photo_b64})

    result = await client.call_tool(
        "analyze_claim_photo",
        {
            "claim_ref": claim_ref, "order_ref": "1", "claim_category": "Wrong Item Received",
            "claim_description": "I ordered a laptop but received this instead.",
        },
    )
    data = result.data
    assert data["verdict"] in ("consistent", "partially_consistent", "inconsistent")
    assert isinstance(data["product_match"], bool)
    assert isinstance(data["reasoning"], str) and len(data["reasoning"]) > 0
    assert data["product_match"] is False, "a plain red square should not be judged to match order 1's real product"


@pytest.mark.parametrize(
    "image_verdict,fraud_risk_band,expected",
    [
        ("consistent", "low", "approve"),
        ("consistent", "medium", "approve"),
        ("consistent", "high", "escalate"),
        ("partially_consistent", "low", "approve"),
        ("partially_consistent", "medium", "escalate"),
        ("partially_consistent", "high", "escalate"),
        ("inconsistent", "low", "escalate"),
        ("inconsistent", "medium", "deny"),
        ("inconsistent", "high", "deny"),
    ],
)
async def test_apply_decision_matrix_covers_full_grid(client: Client, image_verdict, fraud_risk_band, expected):
    """All 9 real cells of docs/refund_policy.md's Decision matrix, not a
    sample -- this tool is a pure lookup, so every cell is cheap to check
    for real rather than trusting a handful of examples."""
    result = await client.call_tool(
        "apply_decision_matrix",
        {"image_verdict": image_verdict, "fraud_risk_band": fraud_risk_band, "refund_amount_usd": 50.0},
    )
    assert result.data["decision"] == expected


async def test_apply_decision_matrix_high_value_always_escalates(client: Client):
    """The >$200 guardrail overrides the matrix outcome even for the
    lowest-risk cell (consistent/low, which would otherwise auto-approve)."""
    result = await client.call_tool(
        "apply_decision_matrix",
        {"image_verdict": "consistent", "fraud_risk_band": "low", "refund_amount_usd": 200.01},
    )
    assert result.data["decision"] == "escalate"


async def test_apply_decision_matrix_rejects_no_photo(client: Client):
    """'no_photo' is a real ConsistencyAssessment/image_consistency value
    elsewhere in the system, but this tool deliberately doesn't accept it --
    that case is Flow-level re-prompt routing, never a Verdict."""
    with pytest.raises(Exception, match="no decision-matrix entry"):
        await client.call_tool(
            "apply_decision_matrix",
            {"image_verdict": "no_photo", "fraud_risk_band": "low", "refund_amount_usd": 50.0},
        )
