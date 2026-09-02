-- Slice 1: only the tables the Orchestrator MCP server's non-LLM tools need.
-- customers is the ONLY table that ever holds a raw customer identifier
-- (email/customer_id) -- per project-plan.md Q27, nothing built on top of
-- this reads that raw column back into any agent's LLM context.

CREATE TABLE IF NOT EXISTS customers (
    customer_ref TEXT PRIMARY KEY,
    raw_identifier TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- project-plan.md Q98: DummyJSON carts have no purchase-date field at all,
-- so there was previously no real value to check a customer's self-reported
-- days_to_return against. get_order's _get_or_seed_order_date bootstrap-
-- samples a stable synthetic order_date (1-60 days before "now") the first
-- time a given order_ref is seen and persists it here -- the same pattern
-- customer_profiles already uses for get_account_info, just keyed on
-- order_ref instead of customer_ref.
CREATE TABLE IF NOT EXISTS order_metadata (
    order_ref TEXT PRIMARY KEY,
    order_date TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS refund_transactions (
    order_ref TEXT NOT NULL,
    claim_ref TEXT NOT NULL,
    amount NUMERIC(10, 2) NOT NULL,
    reason TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (order_ref, claim_ref)
);

-- Fraud Scoring's get_account_info tool needs several account-level features
-- (total_orders_lifetime, total_returns_lifetime, customer_support_contacts_90d,
-- previous_dispute_count, address_match, account_age_days) that no real data
-- source in this project provides -- DummyJSON has no per-customer lifetime
-- history. Seeded once per customer_ref, on first get_account_info call, by
-- bootstrap-sampling a real row from ml/data/synthetic_fraud_risk_dataset.csv
-- (all 6 fields sampled together from the same row to preserve the real
-- correlations between them, rather than drawn independently per column).
CREATE TABLE IF NOT EXISTS customer_profiles (
    customer_ref TEXT PRIMARY KEY REFERENCES customers(customer_ref),
    account_age_days INTEGER NOT NULL,
    total_orders_lifetime INTEGER NOT NULL,
    total_returns_lifetime INTEGER NOT NULL,
    customer_support_contacts_90d INTEGER NOT NULL,
    previous_dispute_count INTEGER NOT NULL,
    address_match BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- project-plan.md Q72: a real claim with outcome='escalate' gets a row
-- here so the Reviewer Dashboard has something queryable to list --
-- before this table, an escalated claim's outcome just returned from the
-- API and nothing remembered it needed human review. 'deny' is a firm
-- auto-deny from the real decision matrix, not queued here (Q72).
-- No raw customer identifier lives here (Q27) -- customer_ref only, same
-- boundary as every other table.
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
);

-- project-plan.md Q100: NULL until the Customer Chat Frontend has actually
-- shown the customer this decision -- POST /claim-updates sets it to now()
-- the same call it returns the decision in, so a reviewer's approve/deny
-- is surfaced to the customer exactly once, not on every subsequent poll.
-- ADD COLUMN IF NOT EXISTS (not just part of CREATE TABLE) so this also
-- migrates an already-initialized live review_queue, not just a fresh one.
ALTER TABLE review_queue ADD COLUMN IF NOT EXISTS customer_notified_at TIMESTAMPTZ;

-- project-plan.md Q96 (admin "All Claims" tab): a real, complete log of
-- every claim ClaimTriageFlow ever finishes running -- not just the
-- outcome='escalate' subset review_queue (Q72) tracks. claim_ref is the
-- primary key and each real backend._run_claim() completion UPSERTs this
-- row (ON CONFLICT DO UPDATE): a claim that first comes back
-- outcome='re_prompt_for_photo' (the customer hasn't sent a photo yet)
-- and is later re-run after the photo arrives updates this same row to
-- its new, current state rather than creating a second row for the same
-- claim_ref. No raw customer identifier lives here (Q27).
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
);

-- project-plan.md Q99: which real DummyJSON orders a customer_ref "owns" --
-- there was previously no ownership concept anywhere in this system
-- (order_ref was a free-typed value any customer could submit for any
-- order). Bootstrap-sampled once per customer_ref (3 real, distinct
-- DummyJSON carts, real product titles/amounts) by
-- _get_or_seed_customer_orders on first use, then stable -- the real data
-- source for the Customer Chat Frontend's "which order is this about?"
-- dropdown.
CREATE TABLE IF NOT EXISTS customer_orders (
    customer_ref TEXT NOT NULL REFERENCES customers(customer_ref),
    order_ref TEXT NOT NULL,
    product_title TEXT NOT NULL,
    amount_usd NUMERIC(10, 2) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_ref, order_ref)
);
