"""FastAPI backend (project-plan.md's Tech stack: "sits between Streamlit
and the CrewAI pipeline, handles request routing and session wiring").

Wraps agents/flow.py's ClaimTriageFlow as a real HTTP endpoint. Two entry
points:

  POST /claims   -- already-structured intake (claim_category/description/
                     days_to_return supplied directly). The original,
                     simpler path -- unchanged from when it was the only one.
  POST /messages -- a raw customer message. May take several round-trips:
                     the Orchestrator's intake Task (agents/orchestrator_
                     agent.py's build_intake_task) extracts what it can and
                     asks a follow_up_question for whatever's missing,
                     persisting partial progress via the real
                     get/put_conversation_state MCP tools (built in Slice 1,
                     unused until this slice) keyed on customer_ref. Once
                     every required field is known, the same real
                     ClaimTriageFlow runs.
  POST /messages/photo -- channel-agnostic photo attachment, for the
                     customer-facing chat frontend (project-plan.md Q78):
                     shares _get_or_mint_claim_ref with the text path so a
                     photo can attach to a claim before every text field
                     is known.

Requires mcp-servers/orchestrator_server.py to already be running (same
requirement as agents/mcp_tools.py) -- this backend doesn't start it.
"""

import base64
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import psycopg
from crewai import Crew
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agents.flow import PHOTO_REQUIRED_CATEGORIES, ClaimState, ClaimTriageFlow, _parse_tool_result
from agents.mcp_tools import (
    DECISION_TOOLS,
    FRAUD_SCORING_TOOLS,
    IMAGE_PARSING_TOOLS,
    ORCHESTRATOR_TOOLS,
    orchestrator_mcp_adapter,
)
from agents.orchestrator_agent import build_intake_task, build_orchestrator_agent
from agents.schemas import ClaimCategory

# Q96: the Eval/Decision Agent's real RAG pipeline (chunk/embed/index into
# OpenSearch Serverless) already lives in ml/rag/ as standalone scripts with
# bare, non-package imports -- mcp-servers/orchestrator_server.py already
# reaches it the same way, via sys.path, rather than duplicating the logic.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml" / "rag"))
from chunk_document import chunk_document  # noqa: E402
from embed_and_index import embed_texts, get_azure_client  # noqa: E402
from index_chunks import ensure_index, get_opensearch_client, index_chunks  # noqa: E402

app = FastAPI(title="Fraud Triage API")

DATABASE_URL = os.environ["DATABASE_URL"]
REVIEWER_PASSWORD = os.environ.get("REVIEWER_PASSWORD")


def require_reviewer(x_reviewer_password: str = Header(default="")) -> None:
    """Q31's shared reviewer password, checked here too -- not just by the
    Streamlit page -- so these endpoints aren't only protected by a UI
    that happens to ask for a password first. Defense in depth, matching
    this project's consistent pattern (e.g. IssueRefundGate)."""
    if not REVIEWER_PASSWORD or x_reviewer_password != REVIEWER_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid or missing reviewer password.")

# project-plan.md Q101: days_to_return dropped -- no longer customer-
# provided at all (see _real_days_to_return below, and orchestrator_
# agent.py's build_intake_task, which no longer asks for it).
_INTAKE_FIELDS = ("order_ref", "claim_category", "claim_description")


class ClaimRequest(BaseModel):
    customer_identifier: str = Field(..., description="Raw email/customer ID -- resolved to customer_ref internally, never returned or logged (Q27).")
    order_ref: str
    claim_category: ClaimCategory
    claim_description: str


class ClaimResponse(BaseModel):
    """Deliberately excludes customer_identifier -- Q27's PII boundary
    applies to API responses too, not just agent LLM context. Every other
    field here is either an opaque reference or a real, grounded result."""

    claim_ref: str
    order_ref: str
    photo_required: bool
    image_verdict: Optional[str] = None
    image_analysis_error: Optional[str] = None
    fraud_risk_band: Optional[str] = None
    decision: Optional[str] = None
    refund_form: Optional[str] = None
    policy_clause: Optional[str] = None
    policy_version: Optional[str] = None
    decision_error: Optional[str] = None
    transaction_id: Optional[str] = None
    outcome: str


# Q39/Q45: 3 claim submissions per customer_ref per hour, in-process, no
# distributed limiter needed at this scale (project-plan.md). NOT slowapi
# (the "e.g." in Q39's answer, not a mandate): slowapi's key function
# extracts synchronously from the raw request, but the rate-limit key here
# is customer_ref, which only exists after a real resolve_customer_ref MCP
# call -- a small in-process, thread-safe counter fits this shape better
# than forcing an async-resolved value through a sync key function.
_RATE_LIMIT_MAX = 3
_RATE_LIMIT_WINDOW_SECONDS = 3600
_rate_limit_lock = threading.Lock()
_rate_limit_history: dict[str, list[float]] = {}


def _check_rate_limit(customer_ref: str) -> None:
    now = time.time()
    with _rate_limit_lock:
        history = [t for t in _rate_limit_history.get(customer_ref, []) if now - t < _RATE_LIMIT_WINDOW_SECONDS]
        if len(history) >= _RATE_LIMIT_MAX:
            _rate_limit_history[customer_ref] = history
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: max {_RATE_LIMIT_MAX} claim submissions per hour.",
            )
        history.append(now)
        _rate_limit_history[customer_ref] = history


def _resolve_customer_ref(customer_identifier: str) -> str:
    """A lightweight, standalone resolve_customer_ref call -- separate from
    ClaimTriageFlow's own resolution inside resolve_customer_and_order --
    specifically so the rate limit (keyed on customer_ref, Q39) can be
    checked BEFORE the expensive Flow (real LLM/vision/MCP calls) ever
    runs. resolve_customer_ref is idempotent, so this costs one extra cheap
    Postgres lookup, not a correctness issue."""
    from agents.mcp_tools import orchestrator_mcp_adapter  # noqa: PLC0415 -- see docstring

    with orchestrator_mcp_adapter() as tools:
        resolve = next(t for t in tools if t.name == "resolve_customer_ref")
        return resolve.run(customer_id_or_email=customer_identifier)


def _order_belongs_to_customer(customer_ref: str, order_ref: str) -> bool:
    """project-plan.md Q101: order_ref may only ever be one of this
    customer's own real, bootstrap-seeded orders (Q99's customer_orders
    table) -- checked directly against Postgres, not just trusted from
    whatever the frontend's dropdown happened to send, since the same
    guarantee has to hold for any caller of these endpoints, not only a
    well-behaved browser session that only ever offers real choices."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM customer_orders WHERE customer_ref = %s AND order_ref = %s",
                (customer_ref, order_ref),
            )
            return cur.fetchone() is not None


def _real_days_to_return(order_ref: str) -> int:
    """project-plan.md Q101: replaces the customer's own self-reported
    "how many days ago" entirely -- once order_ref is confirmed to be a
    real order this customer owns (_order_belongs_to_customer above), its
    real order_date is already known and trustworthy via get_days_to_return,
    so there's no remaining reason to ask and risk a wrong or dishonest
    answer feeding the fraud model and Decision Agent."""
    with orchestrator_mcp_adapter() as tools:
        get_days = next(t for t in tools if t.name == "get_days_to_return")
        return int(get_days.run(order_ref=order_ref))


def _build_claim_response(s: ClaimState) -> ClaimResponse:
    return ClaimResponse(
        claim_ref=s.claim_ref,
        order_ref=s.order_ref,
        photo_required=s.claim_category in PHOTO_REQUIRED_CATEGORIES,
        image_verdict=s.image_verdict or None,
        image_analysis_error=s.image_analysis_error,
        fraud_risk_band=s.fraud_risk_band or None,
        decision=s.decision or None,
        refund_form=s.refund_form,
        policy_clause=s.policy_clause,
        policy_version=s.policy_version,
        decision_error=s.decision_error,
        transaction_id=s.transaction_id,
        outcome=s.outcome,
    )


def _insert_review_queue_row(customer_ref: str, s: ClaimState) -> None:
    """Q72: the only place a review_queue row gets created -- called once,
    right after a real Flow run, only when outcome=='escalate'. Direct
    psycopg, not an MCP tool: this is Flow/backend bookkeeping (like the
    rate limiter above), not an agent's own tool call, so it isn't subject
    to the "every agent tool call goes through MCP" rubric requirement."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO review_queue (claim_ref, order_ref, customer_ref, claim_category, "
                "claim_description, refund_amount_usd, image_verdict, fraud_risk_band, verdict_reasoning) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    s.claim_ref, s.order_ref, customer_ref, s.claim_category, s.claim_description,
                    s.refund_amount_usd, s.image_verdict or None, s.fraud_risk_band or None, s.verdict_reasoning or None,
                ),
            )
        conn.commit()


def _upsert_claim_log_row(customer_ref: str, s: ClaimState) -> None:
    """Q96: unlike `_insert_review_queue_row` above (Q72, only fires for
    outcome=='escalate'), this logs EVERY claim outcome -- called once per
    real `_run_claim()` completion, for the admin "All Claims" tab.
    UPSERTed on claim_ref, not blindly inserted: a claim that first comes
    back outcome='re_prompt_for_photo' and is later re-run after the
    customer sends a photo updates the same row to its new, current state
    rather than creating a second row for the same claim_ref."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO claims (claim_ref, customer_ref, order_ref, claim_category, claim_description, "
                "refund_amount_usd, image_verdict, fraud_risk_band, decision, outcome, transaction_id, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now()) "
                "ON CONFLICT (claim_ref) DO UPDATE SET "
                "refund_amount_usd = EXCLUDED.refund_amount_usd, image_verdict = EXCLUDED.image_verdict, "
                "fraud_risk_band = EXCLUDED.fraud_risk_band, decision = EXCLUDED.decision, "
                "outcome = EXCLUDED.outcome, transaction_id = EXCLUDED.transaction_id, updated_at = now()",
                (
                    s.claim_ref, customer_ref, s.order_ref, s.claim_category, s.claim_description,
                    s.refund_amount_usd, s.image_verdict or None, s.fraud_risk_band or None,
                    s.decision or None, s.outcome, s.transaction_id,
                ),
            )
        conn.commit()


def _run_claim(claim_ref: str, customer_identifier: str, customer_ref: str, order_ref: str, claim_category: str, claim_description: str, days_to_return: int) -> ClaimResponse:
    flow = ClaimTriageFlow()
    flow.kickoff(
        inputs={
            "claim_ref": claim_ref,
            "customer_identifier": customer_identifier,
            "order_ref": order_ref,
            "claim_category": claim_category,
            "claim_description": claim_description,
            "days_to_return": days_to_return,
        }
    )
    _upsert_claim_log_row(customer_ref, flow.state)
    if flow.state.outcome == "escalate":
        _insert_review_queue_row(customer_ref, flow.state)
    return _build_claim_response(flow.state)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tool-registry", dependencies=[Depends(require_reviewer)])
def tool_registry() -> dict:
    """Q71: read-only visibility only -- lists which MCP tools each agent
    is actually wired to call (agents/mcp_tools.py's real per-agent
    constants, not a separately-maintained description that could drift).
    Not a manage/enable-disable console; that heavier scope was never
    asked for."""
    return {
        "Orchestrator": list(ORCHESTRATOR_TOOLS),
        "Image Parsing Agent": list(IMAGE_PARSING_TOOLS),
        "Fraud Scoring Agent": list(FRAUD_SCORING_TOOLS),
        "Decision Agent": list(DECISION_TOOLS),
    }


@app.post("/claims", response_model=ClaimResponse)
def submit_claim(request: ClaimRequest) -> ClaimResponse:
    customer_ref = _resolve_customer_ref(request.customer_identifier)
    if not _order_belongs_to_customer(customer_ref, request.order_ref):
        raise HTTPException(
            status_code=400,
            detail=f"order_ref={request.order_ref!r} is not one of this customer's own orders (project-plan.md Q101).",
        )
    _check_rate_limit(customer_ref)
    claim_ref = f"clm_{uuid.uuid4().hex[:10]}"
    days_to_return = _real_days_to_return(request.order_ref)
    return _run_claim(
        claim_ref, request.customer_identifier, customer_ref, request.order_ref, request.claim_category,
        request.claim_description, days_to_return,
    )


class MessageRequest(BaseModel):
    customer_identifier: str
    message: str
    order_ref: Optional[str] = Field(
        default=None,
        description="Q99: set when the customer picked an order from the frontend's dropdown "
        "(list_customer_orders) instead of describing it in free text -- takes priority over "
        "whatever the intake LLM would otherwise try to extract from the message.",
    )


class MessageResponse(BaseModel):
    needs_more_info: bool
    follow_up_question: Optional[str] = None
    request_type: str
    claim_result: Optional[ClaimResponse] = None


def _get_conversation_state(customer_ref: str) -> dict:
    with orchestrator_mcp_adapter() as tools:
        get_state = next(t for t in tools if t.name == "get_conversation_state")
        return _parse_tool_result(get_state.run(customer_ref=customer_ref)) or {}


def _put_conversation_state(customer_ref: str, state: dict) -> None:
    with orchestrator_mcp_adapter() as tools:
        put_state = next(t for t in tools if t.name == "put_conversation_state")
        put_state.run(customer_ref=customer_ref, state=state)


def _get_or_mint_claim_ref(customer_ref: str, prior_state: dict) -> str:
    """Minted on a claim's first turn and persisted immediately (project-
    plan.md's Twilio/WhatsApp slice, Q76) -- not deferred to Flow-kickoff
    time the way it used to be, since a WhatsApp photo can arrive attached
    to any turn, including before every text field is known, and
    store_photo needs a real claim_ref to attach it to right then, not
    whenever intake happens to finish.

    Only reused while an intake is genuinely still in progress for THIS
    claim (claim_status == 'awaiting_details'); a customer_ref's previous,
    already-resolved claim must never leak its claim_ref into a new one --
    minting fresh resets the persisted state rather than layering onto
    stale fields from that old claim."""
    if prior_state.get("claim_status") == "awaiting_details" and prior_state.get("claim_ref"):
        return prior_state["claim_ref"]
    claim_ref = f"clm_{uuid.uuid4().hex[:10]}"
    _put_conversation_state(customer_ref, {"claim_ref": claim_ref})
    return claim_ref


def _process_message(customer_identifier: str, message: str, order_ref: Optional[str] = None) -> MessageResponse:
    """Shared by POST /messages and the Twilio webhook (Q76) -- one real
    intake pipeline behind both entry points, not a duplicated copy for
    WhatsApp."""
    customer_ref = _resolve_customer_ref(customer_identifier)
    prior_state = _get_conversation_state(customer_ref)

    # Q39 (revised for multi-turn intake): only a genuinely new claim
    # counts against the rate limit -- a message continuing an
    # already-in-progress intake doesn't, so a claim needing several
    # follow-up exchanges isn't penalized for its own back-and-forth.
    is_continuing_intake = prior_state.get("claim_status") == "awaiting_details"
    if not is_continuing_intake:
        _check_rate_limit(customer_ref)

    claim_ref = _get_or_mint_claim_ref(customer_ref, prior_state)
    known_fields = {field: prior_state.get(field) for field in _INTAKE_FIELDS}

    # project-plan.md Q101: order_ref may ONLY ever come from here -- the
    # dropdown-selected value on this turn, or an already-validated value
    # persisted from an earlier turn -- never from the intake LLM's own
    # extraction of the message text (build_intake_task no longer even
    # asks it to try). A customer typing a different order number in free
    # text has no effect on which order the claim is actually filed
    # against.
    if order_ref:
        if not _order_belongs_to_customer(customer_ref, order_ref):
            return MessageResponse(
                needs_more_info=True,
                follow_up_question="That doesn't look like one of your orders. Please pick one from the dropdown above.",
                request_type="new_claim",
            )
        known_fields["order_ref"] = order_ref
    elif known_fields.get("order_ref") and not _order_belongs_to_customer(customer_ref, known_fields["order_ref"]):
        # Defensive re-check of persisted state, not just a first-write
        # check -- cheap (one indexed lookup) and keeps the guarantee an
        # invariant rather than a one-time gate.
        known_fields["order_ref"] = None

    with orchestrator_mcp_adapter() as tools:
        agent = build_orchestrator_agent(tools, customer_ref, claim_ref)
        task = build_intake_task(agent, message, known_fields)
        result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
    intake = result.pydantic

    if intake.request_type == "general_inquiry":
        return MessageResponse(needs_more_info=False, request_type="general_inquiry")

    merged = dict(known_fields)
    for field in ("claim_category", "claim_description"):
        value = getattr(intake, field)
        if value is not None:
            merged[field] = value

    missing = [field for field in _INTAKE_FIELDS if merged.get(field) is None]
    if missing:
        follow_up = (
            "Please select which order this is about from the dropdown above, then tell me what happened."
            if "order_ref" in missing
            else intake.follow_up_question
        )
        _put_conversation_state(customer_ref, {**merged, "claim_status": "awaiting_details", "claim_ref": claim_ref})
        return MessageResponse(
            needs_more_info=True, follow_up_question=follow_up, request_type=intake.request_type,
        )

    days_to_return = _real_days_to_return(merged["order_ref"])
    claim_result = _run_claim(
        claim_ref, customer_identifier, customer_ref, merged["order_ref"], merged["claim_category"],
        merged["claim_description"], days_to_return,
    )
    # re_prompt_for_photo means the claim is NOT done -- the Flow is still
    # waiting on the customer, same as needs_more_info above, just
    # discovered a step later (inside ClaimTriageFlow's photo-required
    # check, not the Orchestrator's own intake-field check). Marking this
    # "resolved" like a genuine final outcome was a real bug: it broke
    # claim_ref continuity for the photo upload and the customer's next
    # message (each minted a fresh, disconnected claim_ref instead of
    # continuing this one) and wrongly charged the follow-up against the
    # rate limit, even though it's the same back-and-forth Q39 already
    # exempts for missing-field follow-ups.
    if claim_result.outcome == "re_prompt_for_photo":
        # Persist the already-known intake fields too (same shape as the
        # missing-fields branch above) -- otherwise the next message loses
        # order_ref/claim_category/etc. and gets re-prompted for them
        # instead of proceeding straight to re-running the claim now that
        # a photo exists.
        _put_conversation_state(customer_ref, {**merged, "claim_status": "awaiting_details", "claim_ref": claim_result.claim_ref})
    else:
        _put_conversation_state(customer_ref, {"claim_status": "resolved", "claim_ref": claim_result.claim_ref})
    return MessageResponse(needs_more_info=False, request_type=intake.request_type, claim_result=claim_result)


@app.post("/messages", response_model=MessageResponse)
def submit_message(request: MessageRequest) -> MessageResponse:
    return _process_message(request.customer_identifier, request.message, request.order_ref)


class OrdersRequest(BaseModel):
    customer_identifier: str


class OrderOption(BaseModel):
    order_ref: str
    product_title: str
    amount_usd: float
    order_date: str
    returnable: bool
    days_since_order: int


@app.post("/orders", response_model=list[OrderOption])
def list_orders(request: OrdersRequest) -> list[OrderOption]:
    """Q99: the Customer Chat Frontend's real data source for the "which
    order is this about?" dropdown -- customer_identifier in the request
    body (never a URL path/query param), same PII-conscious convention
    POST /messages already uses. Resolves to customer_ref server-side and
    calls the real list_customer_orders MCP tool; only that customer's own
    (bootstrap-seeded, Q99) orders come back. Each includes a real
    returnable flag (Q101: real order_date vs. the refund policy's 30-day
    return window), not just the raw order facts -- the seeded set always
    has at least one of each side."""
    customer_ref = _resolve_customer_ref(request.customer_identifier)
    with orchestrator_mcp_adapter() as tools:
        list_orders_tool = next(t for t in tools if t.name == "list_customer_orders")
        orders = _parse_tool_result(list_orders_tool.run(customer_ref=customer_ref)) or []
    return [OrderOption(**o) for o in orders]


class ClaimUpdate(BaseModel):
    claim_ref: str
    order_ref: str
    status: str  # "approved" or "denied" -- review_queue's own vocabulary (Q72), distinct from ClaimResponse.outcome's
    refund_amount_usd: float
    transaction_id: Optional[str] = None


@app.post("/claim-updates", response_model=list[ClaimUpdate])
def get_claim_updates(request: OrdersRequest) -> list[ClaimUpdate]:
    """Q100: how the customer finds out a human reviewer approved/denied
    their escalated claim -- the Reviewer Dashboard's decide endpoint
    doesn't (and, as a Streamlit app with no server-push/websocket, can't)
    notify the customer at the moment a reviewer acts. Instead, the
    Customer Chat Frontend polls this endpoint on every render; a decision
    is only ever returned once -- customer_notified_at is set to now() in
    the same request that returns it, so re-polling (which happens on
    every Streamlit rerun) doesn't re-announce an already-shown decision."""
    customer_ref = _resolve_customer_ref(request.customer_identifier)
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT claim_ref, order_ref, status, refund_amount_usd, transaction_id FROM review_queue "
                "WHERE customer_ref = %s AND status IN ('approved', 'denied') AND customer_notified_at IS NULL",
                (customer_ref,),
            )
            rows = cur.fetchall()
            if rows:
                cur.execute(
                    "UPDATE review_queue SET customer_notified_at = now() WHERE claim_ref = ANY(%s)",
                    ([r[0] for r in rows],),
                )
        conn.commit()
    return [
        ClaimUpdate(claim_ref=r[0], order_ref=r[1], status=r[2], refund_amount_usd=float(r[3]), transaction_id=r[4])
        for r in rows
    ]


class PhotoUploadResponse(BaseModel):
    claim_ref: str


@app.post("/messages/photo", response_model=PhotoUploadResponse)
async def submit_message_photo(customer_identifier: str = Form(...), photo: UploadFile = File(...)) -> PhotoUploadResponse:
    """Channel-agnostic photo attachment (project-plan.md's chat-frontend
    slice) -- shares _get_or_mint_claim_ref with _process_message so a
    photo uploaded before every text field is known still lands on the
    same claim as the eventual POST /messages calls that complete intake."""
    customer_ref = _resolve_customer_ref(customer_identifier)
    prior_state = _get_conversation_state(customer_ref)
    claim_ref = _get_or_mint_claim_ref(customer_ref, prior_state)

    photo_bytes = await photo.read()
    photo_base64 = base64.b64encode(photo_bytes).decode("ascii")
    with orchestrator_mcp_adapter() as tools:
        store_photo = next(t for t in tools if t.name == "store_photo")
        store_photo.run(claim_ref=claim_ref, photo_base64=photo_base64)

    return PhotoUploadResponse(claim_ref=claim_ref)


# --- All Claims (Q96) ----------------------------------------------------


class ClaimLogItem(BaseModel):
    claim_ref: str
    order_ref: str
    claim_category: str
    claim_description: str
    refund_amount_usd: float
    image_verdict: Optional[str] = None
    fraud_risk_band: Optional[str] = None
    decision: Optional[str] = None
    outcome: str
    transaction_id: Optional[str] = None
    created_at: str
    updated_at: str


@app.get("/claims", response_model=list[ClaimLogItem], dependencies=[Depends(require_reviewer)])
def list_claims() -> list[ClaimLogItem]:
    """Q96: every claim ClaimTriageFlow has ever finished running, not just
    review_queue's escalated subset -- the admin "All Claims" tab's real
    data source, backed by `claims` (see infra/init.sql)."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT claim_ref, order_ref, claim_category, claim_description, refund_amount_usd, "
                "image_verdict, fraud_risk_band, decision, outcome, transaction_id, created_at, updated_at "
                "FROM claims ORDER BY updated_at DESC"
            )
            rows = cur.fetchall()
    return [
        ClaimLogItem(
            claim_ref=r[0], order_ref=r[1], claim_category=r[2], claim_description=r[3],
            refund_amount_usd=float(r[4]), image_verdict=r[5], fraud_risk_band=r[6], decision=r[7],
            outcome=r[8], transaction_id=r[9], created_at=r[10].isoformat(), updated_at=r[11].isoformat(),
        )
        for r in rows
    ]


# --- Add Documents (Q96) -------------------------------------------------


class DocumentIngestResponse(BaseModel):
    source_document: str
    chunks_indexed: int


@app.post("/documents", response_model=DocumentIngestResponse, dependencies=[Depends(require_reviewer)])
async def upload_document(document: UploadFile = File(...)) -> DocumentIngestResponse:
    """Q96: real RAG ingestion for the admin "Add Documents" tab, not just
    file storage -- chunks (ml/rag/chunk_document.py, structure-agnostic
    unlike chunk_policy.py's refund-policy-specific parser), embeds (the
    same Azure text-embedding-3-small client ml/rag/embed_and_index.py
    already uses), and indexes into the same real OpenSearch Serverless
    index search_refund_policy already queries -- an uploaded document is
    immediately retrievable by the Decision Agent, not a separate store."""
    if not document.filename or not document.filename.lower().endswith((".md", ".txt")):
        raise HTTPException(status_code=400, detail="Only .md and .txt documents are supported right now.")

    raw = await document.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail="Document must be UTF-8 text.") from e

    chunks = chunk_document(text, source_document=document.filename)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document had no content to index.")

    azure_client = get_azure_client()
    embeddings = embed_texts(azure_client, [c["text"] for c in chunks])
    for chunk, vector in zip(chunks, embeddings):
        chunk["embedding"] = vector

    os_client = get_opensearch_client()
    ensure_index(os_client)
    index_chunks(os_client, chunks)  # real OpenSearch Serverless refresh delay -- this call blocks ~20s

    return DocumentIngestResponse(source_document=document.filename, chunks_indexed=len(chunks))


# --- Reviewer Dashboard (Q20/Q31/Q72) -----------------------------------


class ReviewQueueItem(BaseModel):
    claim_ref: str
    order_ref: str
    claim_category: str
    claim_description: str
    refund_amount_usd: float
    image_verdict: Optional[str] = None
    fraud_risk_band: Optional[str] = None
    verdict_reasoning: Optional[str] = None
    status: str
    created_at: str


class ReviewDecisionRequest(BaseModel):
    approve: bool


class ReviewDecisionResponse(BaseModel):
    claim_ref: str
    status: str
    transaction_id: Optional[str] = None


@app.get("/review-queue", response_model=list[ReviewQueueItem], dependencies=[Depends(require_reviewer)])
def list_review_queue() -> list[ReviewQueueItem]:
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT claim_ref, order_ref, claim_category, claim_description, refund_amount_usd, "
                "image_verdict, fraud_risk_band, verdict_reasoning, status, created_at "
                "FROM review_queue WHERE status = 'pending' ORDER BY created_at ASC"
            )
            rows = cur.fetchall()
    return [
        ReviewQueueItem(
            claim_ref=r[0], order_ref=r[1], claim_category=r[2], claim_description=r[3],
            refund_amount_usd=float(r[4]), image_verdict=r[5], fraud_risk_band=r[6],
            verdict_reasoning=r[7], status=r[8], created_at=r[9].isoformat(),
        )
        for r in rows
    ]


@app.post("/review-queue/{claim_ref}/decide", response_model=ReviewDecisionResponse, dependencies=[Depends(require_reviewer)])
def decide_review_queue_item(claim_ref: str, request: ReviewDecisionRequest) -> ReviewDecisionResponse:
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT order_ref, claim_description, refund_amount_usd, status FROM review_queue WHERE claim_ref = %s",
                (claim_ref,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"No review_queue entry for claim_ref={claim_ref!r}")
            order_ref, claim_description, refund_amount_usd, status = row
            if status != "pending":
                raise HTTPException(status_code=409, detail=f"claim_ref={claim_ref!r} was already resolved (status={status!r})")

            transaction_id = None
            new_status = "approved" if request.approve else "denied"
            if request.approve:
                # Q11's non-bypassable write path: a human reviewer's
                # approval flows through the exact same issue_refund MCP
                # call the Orchestrator itself uses -- the Orchestrator's
                # own token, not a special reviewer-only path.
                with orchestrator_mcp_adapter() as tools:
                    issue_refund = next(t for t in tools if t.name == "issue_refund")
                    result = _parse_tool_result(
                        issue_refund.run(
                            order_ref=order_ref, claim_ref=claim_ref, amount=float(refund_amount_usd),
                            reason=f"Reviewer-approved escalation: {claim_description}",
                        )
                    )
                transaction_id = result.get("transaction_id")

            cur.execute(
                "UPDATE review_queue SET status = %s, resolved_at = now(), transaction_id = %s WHERE claim_ref = %s",
                (new_status, transaction_id, claim_ref),
            )
        conn.commit()
    return ReviewDecisionResponse(claim_ref=claim_ref, status=new_status, transaction_id=transaction_id)
