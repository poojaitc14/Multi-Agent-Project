"""Orchestrator MCP server (Slice 1) -- now THE single MCP server (Q55).

Implements the tool paths that don't depend on Ollama, the fraud model, or
an embeddings service, so they can be built and verified now: resolve_customer_ref,
get_order, get/put_conversation_state, store/get_photo, store_transcript,
issue_refund. See project-plan.md's "Tool List Per Agent" / "MCP architecture"
for the full contract, and Q27 for why every tool below returns customer_ref/
order_ref plus de-identified fields instead of a raw identifier.

Per Q55, the 4 per-agent MCP servers were consolidated into one -- Image
Parsing's, Fraud Scoring's, and the Decision Agent's tools still need to be added to this
same module as those agents get built. Consolidating removed the network-level
isolation that used to keep issue_refund unreachable from anywhere but the
Orchestrator's own server, so IssueRefundGate below replaces it with an
explicit check instead.

DynamoDB and S3 connect to real AWS, not an emulator (project-plan.md Q51 --
LocalStack was dropped). Point CONVERSATION_STATE_TABLE/CLAIM_PHOTOS_BUCKET/
TRANSCRIPTS_BUCKET at dev-prefixed names for local dev so this never touches
the deployed environment's data, and make sure real AWS credentials are
configured (env vars, ~/.aws/credentials, or an IAM role).
"""

import base64
import json
import os
import random
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import cv2
import easyocr
import numpy as np
import pandas as pd
import psycopg
import requests
from boto3.dynamodb.conditions import Attr, Key
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware, MiddlewareContext

load_dotenv()

# Decision Agent's search_refund_policy tool (Q50/Q54/Q56) is implemented for real in
# ml/rag/ -- imported rather than duplicated, since that's already the
# verified implementation (see project-plan.md's Slice 3 status).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml" / "rag"))
from search_refund_policy import search_refund_policy_or_escalate  # noqa: E402

# Fraud Scoring's score_fraud_risk tool (Q7/Q29/Q34/Q58) -- ml/fraud_attribution.py
# is the real, verified implementation (Slice 4). FraudAttributor is built once,
# eagerly, at import time: its first real call pays a ~5-6s SHAP/numba warmup
# cost (measured in Slice 4), and that needs to happen at server startup, not
# on a customer's first claim.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from fraud_attribution import FEATURE_COLS, FraudAttributor  # noqa: E402

_WARMUP_FEATURES = {
    "account_age_days": 0, "total_orders_lifetime": 0, "total_returns_lifetime": 0,
    "claim_frequency_90d": 0, "refund_amount_usd": 0.0, "days_to_return": 0,
    "customer_support_contacts_90d": 0, "previous_dispute_count": 0,
    "address_match": False, "is_high_value_item": False, "photo_evidence_provided": False,
    "claim_category": "Damaged in Transit", "image_consistency": "consistent",
}
assert set(_WARMUP_FEATURES) == set(FEATURE_COLS), "warmup dict must cover every real feature column"

print("Warming up FraudAttributor (loads the registered model, fits the transformer, builds the SHAP explainer)...")
_fraud_attributor = FraudAttributor()
_fraud_attributor.score(_WARMUP_FEATURES)
print("FraudAttributor ready.")

# redact_photo (Image Parsing) -- OpenCV Haar cascade for faces, EasyOCR for
# incidental text (e.g. a shipping label caught in frame), decided over a
# 2nd hosted vision-API-based approach for cost and detector-accuracy
# reasons. Built once at import time, same rationale as FraudAttributor
# above: EasyOCR's Reader() loads (and, on a fresh machine, downloads) its
# detection/recognition models, a real one-time cost that belongs at server
# startup, not on a customer's first claim. verbose=False is required, not
# just quieter output -- EasyOCR's default progress bar prints a Unicode
# block character that crashes with UnicodeEncodeError on Windows' default
# cp1252 console encoding (hit and confirmed while building this).
print("Loading redact_photo's face/text detectors (OpenCV + EasyOCR)...")
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
print("redact_photo detectors ready.")

DATABASE_URL = os.environ["DATABASE_URL"]
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
# project-plan.md Q95: every boto3 client below used to rely entirely on
# botocore's own defaults (60s connect + 60s read, up to several retries
# with backoff) -- unbounded enough that a single real AWS call stuck for
# genuine network reasons could stall the single-worker MCP server for
# minutes, looking from the outside exactly like "the server is dead" even
# though the process itself stays alive and healthy (confirmed for real
# via a CI heartbeat: RSS stable, alive=yes throughout an entire failed
# run). Tighter, explicit bounds -- still generous for real DynamoDB/S3
# calls, which normally complete in well under a second.
_AWS_CLIENT_CONFIG = Config(connect_timeout=10, read_timeout=15, retries={"max_attempts": 2})
CONVERSATION_STATE_TABLE = os.environ["CONVERSATION_STATE_TABLE"]
CLAIM_PHOTOS_BUCKET = os.environ["CLAIM_PHOTOS_BUCKET"]
TRANSCRIPTS_BUCKET = os.environ["TRANSCRIPTS_BUCKET"]
CLAIM_FREQUENCY_TABLE = os.environ["CLAIM_FREQUENCY_TABLE"]
CLAIM_TTL_DAYS = 14  # project-plan.md Q44
CLAIM_FREQUENCY_WINDOW_DAYS = 90  # matches the claim_frequency_90d feature name
ORCHESTRATOR_MCP_TOKEN = os.environ.get("ORCHESTRATOR_MCP_TOKEN")
TRACKINGMORE_API_KEY = os.environ.get("TRACKINGMORE_API_KEY")
TRACKINGMORE_MAX_RETRIES = 3

# Bootstrap-sample pool for get_account_info's seeded synthetic profile (see
# its docstring below) -- loaded once at import time rather than re-reading
# the CSV on every call.
_SYNTHETIC_PROFILE_COLS = [
    "account_age_days", "total_orders_lifetime", "total_returns_lifetime",
    "customer_support_contacts_90d", "previous_dispute_count", "address_match",
]
_SYNTHETIC_PROFILE_POOL = pd.read_csv(
    Path(__file__).resolve().parent.parent / "ml" / "data" / "synthetic_fraud_risk_dataset.csv"
)[_SYNTHETIC_PROFILE_COLS]


# project-plan.md Q98: get_order and get_product_reference each
# independently GET https://dummyjson.com/carts/{order_ref} -- a literal
# duplicate external call within one claim, repeated again on every
# guardrail retry (Fraud Scoring's own guardrail was observed retrying a
# real claim twice in a row). DummyJSON's demo cart/product data doesn't
# change during a session, so a short in-process TTL cache, keyed on the
# real request URL, removes the duplication without risking stale data
# across sessions. Same threading.Lock()-guarded dict pattern
# backend/main.py's own rate limiter already uses, for consistency.
_DUMMYJSON_CACHE_TTL_SECONDS = 300
_dummyjson_cache_lock = threading.Lock()
_dummyjson_cache: dict[str, tuple[float, dict]] = {}


def _dummyjson_get(url: str) -> dict:
    now = time.monotonic()
    with _dummyjson_cache_lock:
        cached = _dummyjson_cache.get(url)
        if cached is not None and now - cached[0] < _DUMMYJSON_CACHE_TTL_SECONDS:
            return cached[1]

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    with _dummyjson_cache_lock:
        _dummyjson_cache[url] = (now, data)
    return data


def _fetch_cart(order_ref: str) -> dict:
    return _dummyjson_get(f"https://dummyjson.com/carts/{order_ref}")


def _fetch_product(product_id) -> dict:
    return _dummyjson_get(f"https://dummyjson.com/products/{product_id}")


# project-plan.md Q98: DummyJSON carts have no purchase-date field at all,
# so there was previously no way for anything in this system to check a
# customer's self-reported days_to_return against a real order timestamp.
# Same bootstrap-and-persist pattern get_account_info already uses for
# customer_profiles (Q59): the first call for a given order_ref samples a
# value ONCE (a uniform-random offset of 1-60 days before "now") and
# persists it in order_metadata; every later call for that same order_ref
# returns the same stored date, so it stays stable across a session
# instead of silently drifting every time get_order is called.
#
# Deliberately NOT wired into score_fraud_risk's actual model inputs --
# agents/fraud_scoring_agent.py's own docstring already documents a real,
# considered decision that days_to_return stays customer-provided (no
# natural per-claim synthetic value to persist the way a per-order or
# per-customer profile can be). This field is additive, real order
# context returned by get_order, not a replacement for that input.
def _get_or_seed_order_date(order_ref: str) -> str:
    with _pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT order_date FROM order_metadata WHERE order_ref = %s", (order_ref,))
            row = cur.fetchone()
            if row:
                return row[0].isoformat()

            order_date = datetime.now(timezone.utc) - timedelta(days=random.randint(1, 60))
            cur.execute(
                "INSERT INTO order_metadata (order_ref, order_date) VALUES (%s, %s)",
                (order_ref, order_date),
            )
        conn.commit()
    return order_date.isoformat()


def _sample_synthetic_profile() -> tuple:
    """One real row's worth of account-level values, sampled together so the
    fields stay internally correlated (e.g. a long-tenured account plausibly
    also has more lifetime orders) rather than drawn independently per
    column, which could produce implausible combinations."""
    row = _SYNTHETIC_PROFILE_POOL.sample(n=1).iloc[0]
    return (
        int(row["account_age_days"]),
        int(row["total_orders_lifetime"]),
        int(row["total_returns_lifetime"]),
        int(row["customer_support_contacts_90d"]),
        int(row["previous_dispute_count"]),
        bool(row["address_match"]),
    )


class IssueRefundGate(Middleware):
    """Q55: with one shared MCP server, network isolation no longer keeps
    Image/Fraud/Decision Agent's clients away from issue_refund -- this does instead.
    Only issue_refund is gated; every other tool passes through untouched,
    so this doesn't require every caller to authenticate for everything,
    just for the one tool that moves money."""

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        if context.message.name == "issue_refund":
            # include_all=True: get_http_headers() excludes "problematic" headers
            # (host, content-length, and -- relevant here -- authorization) unless
            # asked for explicitly.
            headers = get_http_headers(include_all=True) or {}
            auth_header = headers.get("authorization", "")
            token = auth_header[7:] if auth_header.startswith("Bearer ") else None
            if not ORCHESTRATOR_MCP_TOKEN or token != ORCHESTRATOR_MCP_TOKEN:
                raise ToolError("issue_refund is restricted to the Orchestrator's MCP client")
        return await call_next(context)


mcp = FastMCP("Orchestrator MCP")
mcp.add_middleware(IssueRefundGate())


def _pg_connect():
    return psycopg.connect(DATABASE_URL)


def _dynamodb():
    return boto3.resource("dynamodb", region_name=AWS_REGION, config=_AWS_CLIENT_CONFIG)


def _s3():
    return boto3.client("s3", region_name=AWS_REGION, config=_AWS_CLIENT_CONFIG)


def ensure_local_infra() -> None:
    """Idempotent dev-provisioning against real AWS (project-plan.md Q51):
    creates CONVERSATION_STATE_TABLE/CLAIM_PHOTOS_BUCKET/TRANSCRIPTS_BUCKET
    if they don't already exist yet. Point these env vars at dev-prefixed
    names (e.g. conversation-state-dev) so this never touches the deployed
    environment's real tables/buckets. Call once at server startup (and
    from the test suite before it runs). Requires real AWS credentials
    (env vars, ~/.aws/credentials, or an IAM role) to be configured —
    boto3's default credential chain is used, nothing is hardcoded here."""
    ddb_client = boto3.client("dynamodb", region_name=AWS_REGION, config=_AWS_CLIENT_CONFIG)
    existing_tables = ddb_client.list_tables()["TableNames"]
    if CONVERSATION_STATE_TABLE not in existing_tables:
        ddb_client.create_table(
            TableName=CONVERSATION_STATE_TABLE,
            KeySchema=[{"AttributeName": "customer_ref", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "customer_ref", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        ddb_client.get_waiter("table_exists").wait(TableName=CONVERSATION_STATE_TABLE)
        ddb_client.update_time_to_live(
            TableName=CONVERSATION_STATE_TABLE,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
        )

    if CLAIM_FREQUENCY_TABLE not in existing_tables:
        ddb_client.create_table(
            TableName=CLAIM_FREQUENCY_TABLE,
            KeySchema=[
                {"AttributeName": "customer_ref", "KeyType": "HASH"},
                {"AttributeName": "claim_ref", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "customer_ref", "AttributeType": "S"},
                {"AttributeName": "claim_ref", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        ddb_client.get_waiter("table_exists").wait(TableName=CLAIM_FREQUENCY_TABLE)
        ddb_client.update_time_to_live(
            TableName=CLAIM_FREQUENCY_TABLE,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
        )

    # customer_profiles (get_account_info's seeded synthetic data) -- also
    # created via infra/init.sql for a fresh container, but that only runs
    # once against an empty Postgres data volume, so this idempotent DDL
    # covers an already-initialized local Postgres too.
    with _pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS customer_profiles (
                    customer_ref TEXT PRIMARY KEY REFERENCES customers(customer_ref),
                    account_age_days INTEGER NOT NULL,
                    total_orders_lifetime INTEGER NOT NULL,
                    total_returns_lifetime INTEGER NOT NULL,
                    customer_support_contacts_90d INTEGER NOT NULL,
                    previous_dispute_count INTEGER NOT NULL,
                    address_match BOOLEAN NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS review_queue (
                    claim_ref TEXT PRIMARY KEY,
                    order_ref TEXT NOT NULL,
                    customer_ref TEXT NOT NULL REFERENCES customers(customer_ref),
                    claim_category TEXT NOT NULL,
                    claim_description TEXT NOT NULL,
                    refund_amount_usd NUMERIC(10, 2) NOT NULL,
                    image_verdict TEXT,
                    fraud_risk_band TEXT,
                    verdict_reasoning TEXT,
                    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'denied')),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    resolved_at TIMESTAMPTZ,
                    transaction_id TEXT
                )
                """
            )
            # project-plan.md Q100: see infra/init.sql's comment on this
            # same column.
            cur.execute("ALTER TABLE review_queue ADD COLUMN IF NOT EXISTS customer_notified_at TIMESTAMPTZ")
            # project-plan.md Q97 -- also created via infra/init.sql for a
            # fresh container, mirrored here for the same reason
            # customer_profiles/review_queue are: an already-initialized
            # local Postgres volume that predates this table.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    claim_ref TEXT PRIMARY KEY,
                    customer_ref TEXT NOT NULL REFERENCES customers(customer_ref),
                    order_ref TEXT NOT NULL,
                    claim_category TEXT NOT NULL,
                    claim_description TEXT NOT NULL,
                    refund_amount_usd NUMERIC(10, 2) NOT NULL DEFAULT 0,
                    image_verdict TEXT,
                    fraud_risk_band TEXT,
                    decision TEXT,
                    outcome TEXT NOT NULL,
                    transaction_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # project-plan.md Q98: get_order/get_product_reference's
            # bootstrap-and-persist synthetic order_date (see
            # _get_or_seed_order_date's docstring above).
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS order_metadata (
                    order_ref TEXT PRIMARY KEY,
                    order_date TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # project-plan.md Q99: which real DummyJSON orders a customer
            # owns (see _get_or_seed_customer_orders's docstring above).
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS customer_orders (
                    customer_ref TEXT NOT NULL REFERENCES customers(customer_ref),
                    order_ref TEXT NOT NULL,
                    product_title TEXT NOT NULL,
                    amount_usd NUMERIC(10, 2) NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (customer_ref, order_ref)
                )
                """
            )
        conn.commit()

    s3_client = _s3()
    existing_buckets = {b["Name"] for b in s3_client.list_buckets()["Buckets"]}
    for bucket in (CLAIM_PHOTOS_BUCKET, TRANSCRIPTS_BUCKET):
        if bucket not in existing_buckets:
            # S3 quirk: us-east-1 is the one region that must NOT get a
            # LocationConstraint; every other region requires one, or
            # create_bucket fails with IllegalLocationConstraintException.
            if AWS_REGION == "us-east-1":
                s3_client.create_bucket(Bucket=bucket)
            else:
                s3_client.create_bucket(
                    Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
                )


@mcp.tool
def resolve_customer_ref(customer_id_or_email: str) -> str:
    """Looks up (or creates) the opaque customer_ref for a raw
    customer ID/email. This is the ONLY tool that ever touches the
    raw identifier — its return value (a ref, not the identifier)
    is what every other tool and agent prompt uses from here on."""
    with _pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT customer_ref FROM customers WHERE raw_identifier = %s",
                (customer_id_or_email,),
            )
            row = cur.fetchone()
            if row:
                return row[0]
            customer_ref = f"cst_{uuid.uuid4().hex[:10]}"
            cur.execute(
                "INSERT INTO customers (customer_ref, raw_identifier) VALUES (%s, %s)",
                (customer_ref, customer_id_or_email),
            )
        conn.commit()
    return customer_ref


@mcp.tool
def get_order(order_ref: str) -> dict:
    """Product, amount, order_date, delivery_status, category.
    No shipping name/address/payment fields — those are resolved
    internally by issue_refund when the money actually moves.
    Backed by DummyJSON's public cart API, the closest available
    resource to "order" in the free APIs this project committed to.

    order_date (project-plan.md Q98): a real, stable value now, not always
    None -- DummyJSON itself has no purchase-date field, so
    _get_or_seed_order_date bootstrap-samples and persists one the first
    time a given order_ref is seen, the same pattern get_account_info uses
    for customer_profiles. See that function's docstring for why this
    isn't wired into score_fraud_risk's actual model inputs."""
    cart = _fetch_cart(order_ref)
    first_product = cart["products"][0] if cart.get("products") else {}
    return {
        "product": first_product.get("title", "unknown"),
        "amount": cart.get("total", 0),
        "order_date": _get_or_seed_order_date(order_ref),
        "delivery_status": "delivered",  # DummyJSON has no shipment status; static for this slice
        "category": None,  # set by the claim, not the order — filled in by the Orchestrator agent
    }


# project-plan.md Q99: a real DummyJSON cart ID range, checked for real
# (GET https://dummyjson.com/carts?limit=0 -> {"total": 208}), not
# assumed -- earlier code in this project had assumed a much smaller
# range. Sampling from 1..208 with a retry loop (rather than a fixed
# curated list) means it stays correct even if DummyJSON's own seed data
# size changes.
_DUMMYJSON_CART_ID_MAX = 208
_CUSTOMER_ORDERS_PER_CUSTOMER = 3
# project-plan.md Q99 (revised): DummyJSON's real carts range from a few
# dollars to over $130,000 (some bundle high-quantity motorcycles/jewelry) --
# realistic for a returns-fraud demo means keeping seeded orders in a
# plausible everyday-purchase range. Checked for real, not guessed: a random
# 40-cart sample came back 15/40 (~37%) under $1000, so a generous retry
# budget (20x, not 5x) is needed to reliably find 3 qualifying carts per
# customer without silently under-filling someone's order list.
_MAX_ORDER_AMOUNT_USD = 1000
# project-plan.md Q101: the refund policy's real "Return window" clause
# (docs/refund_policy.md) -- standard claims must be filed within 30 days
# of delivery. Used here as the real, deterministic boundary for whether a
# seeded order counts as "returnable" -- a claim's actual category-specific
# window (e.g. Change of Mind's shorter 14-day sub-window) is still applied
# by the real Decision Agent once a category is chosen, since that's not
# knowable at order-seeding time.
_RETURN_WINDOW_DAYS = 30


def _days_since_order(order_date_iso: str) -> int:
    order_date = datetime.fromisoformat(order_date_iso)
    return (datetime.now(timezone.utc) - order_date).days


def _is_returnable(order_date_iso: str) -> bool:
    return _days_since_order(order_date_iso) <= _RETURN_WINDOW_DAYS


def _sample_qualifying_order(accepted_refs: set, want_returnable, max_attempts: int):
    """project-plan.md Q101: one candidate DummyJSON cart passing the real
    amount filter (Q99) and, when want_returnable isn't None, matching the
    requested side of the real 30-day return window -- used to guarantee a
    customer's seeded orders include at least one of each (see
    _get_or_seed_customer_orders). Rejected candidates are NOT added to
    accepted_refs, so a cart rejected here for being the wrong side of the
    window is still fair game for a later call wanting that exact side --
    accepted_refs only tracks orders actually placed in the final set, to
    avoid the same order_ref being "won" twice."""
    for _ in range(max_attempts):
        candidate = str(random.randint(1, _DUMMYJSON_CART_ID_MAX))
        if candidate in accepted_refs:
            continue
        try:
            cart = _fetch_cart(candidate)
        except requests.HTTPError:
            continue
        first_product = cart["products"][0] if cart.get("products") else {}
        title = first_product.get("title")
        amount = cart.get("total", 0)
        if not title or amount >= _MAX_ORDER_AMOUNT_USD:
            continue
        order_date_iso = _get_or_seed_order_date(candidate)
        returnable = _is_returnable(order_date_iso)
        if want_returnable is not None and returnable != want_returnable:
            continue
        return (candidate, title, amount, order_date_iso, returnable)
    return None


def _get_or_seed_customer_orders(customer_ref: str) -> list[dict]:
    """project-plan.md Q99/Q101: bootstrap-and-persist which real DummyJSON
    orders a given customer_ref "owns" -- the same pattern
    get_account_info (Q59) and _get_or_seed_order_date (Q98) already
    established, just for order ownership instead of account/order
    metadata. There was previously no ownership concept anywhere in this
    system: order_ref was a free-typed value any customer could submit for
    any order.

    The first call for a given customer_ref samples exactly
    _CUSTOMER_ORDERS_PER_CUSTOMER orders, guaranteeing at least one
    "returnable" (real order_date within the last _RETURN_WINDOW_DAYS) and
    at least one "not returnable" (outside it) -- two dedicated sampling
    phases secure one of each first, then any remaining slot is filled with
    an unconstrained draw. Persists the winners; every later call returns
    the same stored order_refs, but re-derives returnable/days_since_order
    fresh from the real, unchanging order_date each time (never a stored,
    staling boolean) -- an order genuinely can cross the 30-day line
    between two calls, purely because time passed."""
    with _pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT co.order_ref, co.product_title, co.amount_usd, om.order_date "
                "FROM customer_orders co JOIN order_metadata om ON om.order_ref = co.order_ref "
                "WHERE co.customer_ref = %s ORDER BY co.order_ref",
                (customer_ref,),
            )
            rows = cur.fetchall()
            if rows:
                return [
                    {
                        "order_ref": r[0], "product_title": r[1], "amount_usd": float(r[2]),
                        "order_date": r[3].isoformat(), "returnable": _is_returnable(r[3].isoformat()),
                        "days_since_order": _days_since_order(r[3].isoformat()),
                    }
                    for r in rows
                ]

            accepted_refs: set[str] = set()
            accepted: list[tuple] = []
            budget = _CUSTOMER_ORDERS_PER_CUSTOMER * 20

            returnable_pick = _sample_qualifying_order(accepted_refs, True, budget)
            if returnable_pick:
                accepted.append(returnable_pick)
                accepted_refs.add(returnable_pick[0])

            not_returnable_pick = _sample_qualifying_order(accepted_refs, False, budget)
            if not_returnable_pick:
                accepted.append(not_returnable_pick)
                accepted_refs.add(not_returnable_pick[0])

            while len(accepted) < _CUSTOMER_ORDERS_PER_CUSTOMER:
                extra = _sample_qualifying_order(accepted_refs, None, budget)
                if extra is None:
                    break
                accepted.append(extra)
                accepted_refs.add(extra[0])

            for order_ref, title, amount, _order_date_iso, _returnable in accepted:
                cur.execute(
                    "INSERT INTO customer_orders (customer_ref, order_ref, product_title, amount_usd) "
                    "VALUES (%s, %s, %s, %s)",
                    (customer_ref, order_ref, title, amount),
                )
        conn.commit()
    return [
        {
            "order_ref": o, "product_title": t, "amount_usd": float(a),
            "order_date": d, "returnable": r, "days_since_order": _days_since_order(d),
        }
        for o, t, a, d, r in accepted
    ]


@mcp.tool
def get_days_to_return(order_ref: str) -> int:
    """project-plan.md Q101: the real, current elapsed days since this
    order's real (bootstrap-seeded, Q98) order_date -- the actual value
    backend/main.py now uses as the Fraud Scoring/Decision Agent input
    that used to be the customer's own self-reported estimate. Replaces
    that self-report entirely rather than merely cross-checking it: once
    an order_ref is confirmed to be one of the customer's own real orders
    (Q99's ownership enforcement), its order_date is already known and
    trustworthy, so there's no remaining reason to ask the customer and
    risk a wrong or dishonest answer."""
    order_date_iso = _get_or_seed_order_date(order_ref)
    return _days_since_order(order_date_iso)


@mcp.tool
def list_customer_orders(customer_ref: str) -> list[dict]:
    """The Customer Chat Frontend's real data source for the "which order
    is this about?" dropdown (project-plan.md Q99) -- returns the specific
    orders THIS customer_ref owns, not every DummyJSON cart. See
    _get_or_seed_customer_orders's docstring for how ownership is
    established (bootstrap-sampled once, then stable)."""
    return _get_or_seed_customer_orders(customer_ref)


@mcp.tool
def get_conversation_state(customer_ref: str) -> dict:
    """Includes claim_status: 'awaiting_photo' | 'assessing' |
    'escalated' | 'resolved', plus the item's TTL."""
    table = _dynamodb().Table(CONVERSATION_STATE_TABLE)
    result = table.get_item(Key={"customer_ref": customer_ref})
    return result.get("Item", {})


@mcp.tool
def put_conversation_state(customer_ref: str, state: dict, ttl_days: int = CLAIM_TTL_DAYS) -> None:
    """Writes state with a DynamoDB TTL (14 days, confirmed — Q44) —
    a claim stuck at 'awaiting_photo' expires instead of persisting
    forever."""
    table = _dynamodb().Table(CONVERSATION_STATE_TABLE)
    item = {"customer_ref": customer_ref, "ttl": int(time.time()) + ttl_days * 86400, **state}
    table.put_item(Item=item)


@mcp.tool
def store_photo(claim_ref: str, photo_base64: str) -> str:
    """photo_base64 is the raw image bytes, base64-encoded — MCP's
    JSON-RPC transport can't carry arbitrary binary directly, so
    every binary payload in this system crosses the tool boundary
    as base64 (same convention used by get_photo below)."""
    key = f"{claim_ref}.jpg"
    _s3().put_object(Bucket=CLAIM_PHOTOS_BUCKET, Key=key, Body=base64.b64decode(photo_base64))
    return key


@mcp.tool
def get_photo(claim_ref: str) -> str:
    """Returns the photo as base64-encoded bytes (see store_photo)."""
    key = f"{claim_ref}.jpg"
    obj = _s3().get_object(Bucket=CLAIM_PHOTOS_BUCKET, Key=key)
    return base64.b64encode(obj["Body"].read()).decode("ascii")


@mcp.tool
def store_transcript(customer_ref: str, transcript: str) -> None:
    key = f"{customer_ref}/{datetime.now(timezone.utc).isoformat()}.txt"
    _s3().put_object(Bucket=TRANSCRIPTS_BUCKET, Key=key, Body=transcript.encode("utf-8"))


@mcp.tool
def issue_refund(order_ref: str, claim_ref: str, amount: float, reason: str) -> dict:
    """Issue a refund. Callers must have already received a Decision
    Agent 'approve' verdict (or a human reviewer override) — this
    tool does not check that itself, the Orchestrator agent does
    (not yet built in this slice).

    Idempotent: keyed on (order_ref, claim_ref) — checks Postgres for
    an existing refund transaction under that key before writing, and
    returns the existing result on a duplicate call instead of
    refunding twice."""
    with _pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT transaction_id, amount, reason FROM refund_transactions "
                "WHERE order_ref = %s AND claim_ref = %s",
                (order_ref, claim_ref),
            )
            row = cur.fetchone()
            if row:
                return {"status": "already_refunded", "transaction_id": row[0]}

            transaction_id = f"rfd_{uuid.uuid4().hex[:10]}"
            cur.execute(
                "INSERT INTO refund_transactions (order_ref, claim_ref, amount, reason, transaction_id) "
                "VALUES (%s, %s, %s, %s, %s)",
                (order_ref, claim_ref, amount, reason, transaction_id),
            )
        conn.commit()
    return {"status": "refunded", "transaction_id": transaction_id}


@mcp.tool
def get_account_info(customer_ref: str) -> dict:
    """account_age_days, total_orders_lifetime, total_returns_lifetime,
    customer_support_contacts_90d, previous_dispute_count, address_match --
    no name/email/phone (Q27).

    Deliberately synthetic, seeded (project-plan.md, Fraud Scoring tools
    revision): DummyJSON has no per-customer lifetime order/return/support
    history, so there is no real data source for these signals in this
    demo. The first call for a given customer_ref bootstrap-samples one
    real row from ml/data/synthetic_fraud_risk_dataset.csv (all 6 fields
    together, to keep them correlated) and persists it in customer_profiles;
    every later call for that customer_ref returns the same stored values,
    so a given demo customer's risk profile stays stable across a session."""
    with _pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT account_age_days, total_orders_lifetime, total_returns_lifetime, "
                "customer_support_contacts_90d, previous_dispute_count, address_match "
                "FROM customer_profiles WHERE customer_ref = %s",
                (customer_ref,),
            )
            row = cur.fetchone()
            if not row:
                cur.execute("SELECT 1 FROM customers WHERE customer_ref = %s", (customer_ref,))
                if not cur.fetchone():
                    raise ToolError(f"unknown customer_ref: {customer_ref}")

                sampled = _sample_synthetic_profile()
                cur.execute(
                    "INSERT INTO customer_profiles (customer_ref, account_age_days, "
                    "total_orders_lifetime, total_returns_lifetime, "
                    "customer_support_contacts_90d, previous_dispute_count, address_match) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (customer_ref, *sampled),
                )
                conn.commit()
                row = sampled

    account_age_days, total_orders_lifetime, total_returns_lifetime, customer_support_contacts_90d, previous_dispute_count, address_match = row
    return {
        "account_age_days": account_age_days,
        "total_orders_lifetime": total_orders_lifetime,
        "total_returns_lifetime": total_returns_lifetime,
        "customer_support_contacts_90d": customer_support_contacts_90d,
        "previous_dispute_count": previous_dispute_count,
        "address_match": bool(address_match),
    }


@mcp.tool
def get_claim_frequency(customer_ref: str) -> dict:
    """claim_frequency_90d -- a real, sliding 90-day count from
    CLAIM_FREQUENCY_TABLE (DynamoDB), not simulated. Filters by claimed_at
    itself rather than trusting the table's TTL-based deletion alone --
    DynamoDB TTL deletion can lag up to ~48h behind an item's actual expiry,
    so an item just past the 90-day mark might still physically be present."""
    table = _dynamodb().Table(CLAIM_FREQUENCY_TABLE)
    cutoff = int(time.time()) - CLAIM_FREQUENCY_WINDOW_DAYS * 86400
    result = table.query(
        KeyConditionExpression=Key("customer_ref").eq(customer_ref),
        FilterExpression=Attr("claimed_at").gte(cutoff),
        Select="COUNT",
    )
    return {"claim_frequency_90d": result["Count"]}


@mcp.tool
def increment_claim_frequency(customer_ref: str, claim_ref: str) -> None:
    """Records this claim in the sliding-window counter (DynamoDB write).
    Call once per new claim -- PutItem on the same (customer_ref, claim_ref)
    key overwrites in place rather than double-counting a retry/follow-up
    on the same claim_ref. Item TTL is 90 days, matching the window."""
    table = _dynamodb().Table(CLAIM_FREQUENCY_TABLE)
    now = int(time.time())
    table.put_item(
        Item={
            "customer_ref": customer_ref,
            "claim_ref": claim_ref,
            "claimed_at": now,
            "ttl": now + CLAIM_FREQUENCY_WINDOW_DAYS * 86400,
        }
    )


@mcp.tool
def get_tracking_status(order_ref: str) -> dict:
    """delivery_status from a real TrackingMore call (Q24: retry with
    backoff, graceful error rather than a crash) -- NOT address_match.
    address_match moved to get_account_info (see its docstring): DummyJSON
    has no address data at all, so there is nothing here to compare a
    TrackingMore result against, and the seeded synthetic profile is keyed
    by customer_ref, which this tool doesn't take as a parameter.

    order_ref is a DummyJSON cart ID, not a real carrier tracking number,
    and TrackingMore also requires a courier_code this project has no
    source for -- so a "not found"/validation-error response, and therefore
    delivery_status: "unknown", is the expected common case in this demo,
    not a bug. TrackingMore's exact response shape is unverified against a
    live account (no API key configured/tested at the time this was
    written) -- parsing here is defensive and falls back to "unknown"
    rather than assuming a specific field path is correct."""
    if not TRACKINGMORE_API_KEY:
        return {"delivery_status": "unknown"}

    for attempt in range(TRACKINGMORE_MAX_RETRIES):
        try:
            response = requests.post(
                "https://api.trackingmore.com/v4/trackings/realtime",
                headers={"Tracking-Api-Key": TRACKINGMORE_API_KEY},
                json={"tracking_number": order_ref},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            items = data.get("data", {}).get("items") or data.get("data") or []
            if isinstance(items, list) and items:
                status = items[0].get("delivery_status") or items[0].get("status")
                return {"delivery_status": status or "unknown"}
            return {"delivery_status": "unknown"}
        except Exception:  # noqa: BLE001 -- any failure here degrades to "unknown", never a crash
            if attempt < TRACKINGMORE_MAX_RETRIES - 1:
                time.sleep(2**attempt)

    return {"delivery_status": "unknown"}


@mcp.tool
def redact_photo(photo_base64: str) -> str:
    """Auto-detects and blurs incidental faces (OpenCV Haar cascade) and
    text (EasyOCR, e.g. a shipping label caught in frame) in a photo.
    Runs before the photo is handed to either vision model (Q27/Q30's PII
    boundary is 'under any circumstances', not just for the hosted model) --
    returns a redacted copy, base64-encoded, same convention as store_photo/
    get_photo. A photo with no detected faces/text is returned unchanged."""
    try:
        raw_bytes = base64.b64decode(photo_base64, validate=True)
    except Exception as e:
        raise ToolError(f"photo_base64 is not valid base64: {e}") from e
    image = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ToolError("photo_base64 could not be decoded as an image")

    def _blur_region(x: int, y: int, w: int, h: int) -> None:
        x, y = max(x, 0), max(y, 0)
        roi = image[y : y + h, x : x + w]
        if roi.size == 0:
            return
        image[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (51, 51), 30)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for x, y, w, h in _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5):
        _blur_region(x, y, w, h)

    for bbox, _text, _confidence in _ocr_reader.readtext(image):
        x, y, w, h = cv2.boundingRect(np.array(bbox, dtype=np.int32))
        _blur_region(x, y, w, h)

    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ToolError("failed to re-encode the redacted photo")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@mcp.tool
def get_product_reference(order_ref: str) -> dict:
    """title, description, reference_image_url for the first product on the
    order (DummyJSON /carts + /products) -- what analyze_image cross-checks
    the claim photo against. Shares _fetch_cart's cache with get_order
    (project-plan.md Q98): within one claim, both tools were independently
    re-fetching the exact same DummyJSON cart -- a real, confirmed
    duplicate call, not just a theoretical one."""
    cart = _fetch_cart(order_ref)
    first_product = cart["products"][0] if cart.get("products") else {}
    product_id = first_product.get("id")
    if product_id is None:
        return {"title": "unknown", "description": None, "reference_image_url": None}

    product = _fetch_product(product_id)
    return {
        "title": product.get("title", "unknown"),
        "description": product.get("description"),
        "reference_image_url": product.get("thumbnail"),
    }


_CONSISTENCY_JSON_SCHEMA = {
    "name": "consistency_assessment",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["consistent", "partially_consistent", "inconsistent"],
            },
            "product_match": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "product_match", "reasoning"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _azure_chat_client():
    from openai import AzureOpenAI  # noqa: PLC0415 -- only needed by analyze_image

    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        timeout=60.0,  # project-plan.md Q95 -- was unbounded (SDK default: 10 minutes)
    )


@mcp.tool
def analyze_image(
    photo_base64: str, claim_category: str, claim_description: str, reference_title: str, reference_description: str
) -> dict:
    """The vision judgment itself -- a real Azure OpenAI GPT-4.1 mini call,
    not simulated. Always call on redact_photo's output, never a raw photo
    upload (Q27/Q30 PII boundary). Only 'no_photo' from ConsistencyAssessment
    is missing here on purpose -- the calling agent handles that case itself
    before ever reaching this tool, since there is no image to analyze then.

    Only GPT-4.1 mini is wired right now. Llama 3.2 Vision 11B is part of
    the design (Q47, project-plan.md's model-selector) but needs Ollama,
    which isn't installed in this environment yet -- a separate follow-up,
    not silently substituted here.

    Returns {"verdict": "consistent"|"partially_consistent"|"inconsistent",
    "product_match": bool, "reasoning": str}, constrained via Azure's
    JSON-schema response_format (Layer 1 of the guardrail design)."""
    client = _azure_chat_client()
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

    prompt = (
        f"A customer filed a '{claim_category}' return claim, describing the issue as: "
        f'"{claim_description}". The ordered product is "{reference_title}": {reference_description}\n\n'
        "Judge whether the attached photo is consistent with this claim and matches the ordered "
        "product. consistent = the photo clearly shows the ordered product and clearly supports the "
        "claimed issue; partially_consistent = the product matches but the claimed issue isn't clearly "
        "visible (or vice versa); inconsistent = the photo shows a different product, or contradicts "
        "the claim (e.g. no visible damage for a damage claim)."
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{photo_base64}"}},
                ],
            }
        ],
        response_format={"type": "json_schema", "json_schema": _CONSISTENCY_JSON_SCHEMA},
    )
    return json.loads(response.choices[0].message.content)


@mcp.tool
def analyze_claim_photo(claim_ref: str, order_ref: str, claim_category: str, claim_description: str) -> dict:
    """Collapses get_photo -> redact_photo -> get_product_reference ->
    analyze_image into one server-side call (project-plan.md Q86/Q87) -- the
    LLM never sees or retypes the raw photo bytes at all. Fixes a real,
    confirmed bug: GPT-4.1 mini can't reliably reproduce a ~20K-character
    base64 photo blob verbatim as a tool-call argument (verified by
    comparing what it actually sent to redact_photo against the real photo
    byte-for-byte -- the model copies correctly for a while, then silently
    drifts into a plausible-looking but fabricated ending). Retries can't
    fix that kind of long-string generation drift, since it's systematic,
    not a one-off glitch -- the only real fix is keeping the blob out of
    the LLM's own generated arguments entirely, the same principle this
    project already applies to every other PII/binary payload.

    get_photo/redact_photo/get_product_reference/analyze_image remain real,
    independently callable/testable tools in their own right (each already
    has direct test coverage) -- this doesn't replace or remove them, it's
    a second, safer entry point the Image Parsing Agent uses instead of
    orchestrating the same 4 steps itself.

    Returns exactly the shape ConsistencyAssessment needs: {"verdict",
    "product_match", "reasoning"}. verdict='no_photo' (product_match=False)
    when no photo exists for this claim, decided deterministically here --
    no LLM call happens in that case, same as the Task-level check the
    Image Parsing Agent used to do itself before this tool existed."""
    try:
        photo_base64 = get_photo(claim_ref)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "NoSuchKey":
            raise
        return {
            "verdict": "no_photo",
            "product_match": False,
            "reasoning": f"No photo was found for claim {claim_ref}.",
        }

    redacted_base64 = redact_photo(photo_base64)
    reference = get_product_reference(order_ref)
    return analyze_image(
        photo_base64=redacted_base64,
        claim_category=claim_category,
        claim_description=claim_description,
        reference_title=reference["title"],
        reference_description=reference.get("description") or "",
    )


@mcp.tool
def search_refund_policy(query: str) -> dict:
    """Decision Agent's RAG lookup (Q50/Q54/Q56) -- real k-NN search against
    OpenSearch Serverless, real Azure OpenAI embeddings, no mocking. Not
    gated by IssueRefundGate (Q56: read-only, no PII, moves no money).

    Returns {"confident": True, "chunks": [...]} with the top-5 relevant
    policy chunks, or {"confident": False, "action": "escalate", ...} if
    nothing scored above the calibrated confidence threshold (0.62) -- the
    Decision Agent branches on "confident" rather than reasoning from a weak
    match. Requires the OpenSearch Serverless collection to actually exist;
    see ml/rag/provision_opensearch.py if it's been torn down."""
    return search_refund_policy_or_escalate(query)


@mcp.tool
def score_fraud_risk(
    account_age_days: int,
    total_orders_lifetime: int,
    total_returns_lifetime: int,
    claim_frequency_90d: int,
    refund_amount_usd: float,
    days_to_return: int,
    customer_support_contacts_90d: int,
    previous_dispute_count: int,
    address_match: bool,
    is_high_value_item: bool,
    photo_evidence_provided: bool,
    claim_category: str,
    image_consistency: str,
) -> dict:
    """Fraud Scoring Agent's tool (Q7/Q29/Q34/Q58) -- real, calibrated,
    registered model (ml/fraud_attribution.py, Slice 2 + 4), not mocked.

    13 flat, individually-typed parameters -- NOT a single features: dict
    (that was the original design; changed while wiring the real CrewAI
    agent, see project-plan.md's Slice 7 status). A generic dict parameter
    gives FastMCP's auto-generated JSON schema no per-field structure, and
    that was empirically fatal: through real CrewAI tool-calling (Azure
    GPT-4.1 mini via litellm), the agent sent an EMPTY features dict on
    every one of 15 real attempts across 3 separate test runs -- including
    with a literal example JSON object spelled out in the task prompt. Flat
    named parameters give the model a real per-field schema to fill in
    instead of an opaque blob, which tool-calling models handle far more
    reliably.

    The Fraud Scoring Agent assembles these from its other tool calls
    (get_order, get_account_info, get_claim_frequency, get_tracking_status).

    Returns {"risk_band": "low"|"medium"|"high", "risk_score": float,
    "key_signals": [...]}. risk_band is the model's own direct 3-class
    prediction, not a thresholded score (Q58); key_signals is grounded in
    real SHAP values against the actual model (Q34), not agent-invented."""
    features = {
        "account_age_days": account_age_days,
        "total_orders_lifetime": total_orders_lifetime,
        "total_returns_lifetime": total_returns_lifetime,
        "claim_frequency_90d": claim_frequency_90d,
        "refund_amount_usd": refund_amount_usd,
        "days_to_return": days_to_return,
        "customer_support_contacts_90d": customer_support_contacts_90d,
        "previous_dispute_count": previous_dispute_count,
        "address_match": address_match,
        "is_high_value_item": is_high_value_item,
        "photo_evidence_provided": photo_evidence_provided,
        "claim_category": claim_category,
        "image_consistency": image_consistency,
    }
    result = _fraud_attributor.score(features)
    return {k: v for k, v in result.items() if not k.startswith("_")}


_DECISION_MATRIX = {
    ("consistent", "low"): "approve",
    ("consistent", "medium"): "approve",
    ("consistent", "high"): "escalate",
    ("partially_consistent", "low"): "approve",
    ("partially_consistent", "medium"): "escalate",
    ("partially_consistent", "high"): "escalate",
    ("inconsistent", "low"): "escalate",
    ("inconsistent", "medium"): "deny",
    ("inconsistent", "high"): "deny",
}
HIGH_VALUE_ESCALATION_THRESHOLD_USD = 200.0


@mcp.tool
def apply_decision_matrix(image_verdict: str, fraud_risk_band: str, refund_amount_usd: float) -> dict:
    """Deterministic decision-matrix lookup (docs/refund_policy.md's
    "Decision matrix" + its >$200-always-escalate guardrail) -- NOT an LLM
    judgment call. A real MCP tool for the same reason score_fraud_risk and
    analyze_image are real tools rather than agent guesses: the
    approve/deny/escalate label is a pure function of 3 known inputs, so
    grounding it here means the Decision Agent's real contribution is
    picking refund_form/policy_clause/reasoning from retrieved policy text,
    not re-deriving a lookup table an LLM could get wrong.

    image_verdict: "consistent" | "partially_consistent" | "inconsistent"
    (NOT "no_photo" -- a no-photo-on-a-photo-required-category claim never
    reaches this tool or the Decision Agent at all; project-plan.md's
    matrix has that case re-prompting the customer for a photo instead of
    producing a Verdict, which is Flow-level routing logic, decided before
    this point, not something apply_decision_matrix or the Decision Agent's
    Verdict schema -- approve/deny/escalate only -- can express).
    fraud_risk_band: "low" | "medium" | "high"
    refund_amount_usd: the amount being claimed

    Returns {"decision": "approve"|"deny"|"escalate"}. The >$200 guardrail
    is checked first and overrides the matrix outcome either way."""
    if refund_amount_usd > HIGH_VALUE_ESCALATION_THRESHOLD_USD:
        return {"decision": "escalate"}

    decision = _DECISION_MATRIX.get((image_verdict, fraud_risk_band))
    if decision is None:
        raise ToolError(
            f"no decision-matrix entry for image_verdict={image_verdict!r}, "
            f"fraud_risk_band={fraud_risk_band!r} -- image_verdict must be one of "
            "'consistent'/'partially_consistent'/'inconsistent' (not 'no_photo', see docstring) "
            "and fraud_risk_band must be one of 'low'/'medium'/'high'"
        )
    return {"decision": decision}


if __name__ == "__main__":
    ensure_local_infra()
    # host="0.0.0.0" is required for Docker (project-plan.md's Dockerfiles/
    # chat-frontend slices) -- without it FastMCP defaults to 127.0.0.1
    # (confirmed via its own real startup log: "Uvicorn running on
    # http://127.0.0.1:8001"), which only accepts connections from inside
    # this exact container, not from sibling containers on the same Docker
    # network (backend's real "Connection refused" reaching mcp-server:8001
    # is what surfaced this -- DNS resolved fine, the process just wasn't
    # listening on any interface but loopback). Binding 0.0.0.0 is safe for
    # non-Docker local dev too -- it's a superset of 127.0.0.1, not a
    # narrower/different binding.
    mcp.run(transport="http", host="0.0.0.0", port=int(os.environ.get("MCP_SERVER_PORT", 8001)))
