"""Admin app (project-plan.md Q71/Q78): the internal-only Streamlit scope,
separate from chat/app.py's customer-facing chat frontend. Two sections,
both behind the same shared reviewer password (Q31):

  - Tool registry: read-only visibility into which MCP tools each agent
    is wired to call -- calls the real GET /tool-registry endpoint, which
    itself reads agents/mcp_tools.py's real per-agent constants, so this
    can never drift from what's actually wired.
  - Reviewer Dashboard (Q20): lists claims with outcome='escalate' from
    the real review_queue table (Q72) and lets a human approve (issuing
    the refund through the same gated issue_refund path the Orchestrator
    itself uses) or deny each one.

Requires backend/main.py (FastAPI) already running -- this app is a thin
UI over its real endpoints, no direct DB/MCP access of its own.

Run with: uv run streamlit run admin/app.py
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8002")

st.set_page_config(page_title="Fraud Triage Admin", layout="wide")


def _auth_headers() -> dict:
    return {"X-Reviewer-Password": st.session_state.get("reviewer_password", "")}


def _password_gate() -> bool:
    """Q31: nothing below this renders until the shared reviewer password
    is entered -- checked against the real backend (GET /tool-registry is
    the cheapest real endpoint behind require_reviewer), not just compared
    to a value this page happens to know, so a wrong/stale password fails
    the same way it would against any other protected endpoint."""
    if st.session_state.get("reviewer_authenticated"):
        return True

    st.title("Fraud Triage Admin")
    password = st.text_input("Reviewer password", type="password")
    if st.button("Sign in") and password:
        st.session_state["reviewer_password"] = password
        check = requests.get(f"{BACKEND_URL}/tool-registry", headers=_auth_headers(), timeout=10)
        if check.status_code == 200:
            st.session_state["reviewer_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def _render_tool_registry() -> None:
    st.subheader("Tool Registry")
    st.caption("Read-only -- which real MCP tools each agent is wired to call.")
    response = requests.get(f"{BACKEND_URL}/tool-registry", headers=_auth_headers(), timeout=10)
    response.raise_for_status()
    registry = response.json()
    for agent_name, tools in registry.items():
        with st.container(border=True):
            st.markdown(f"**{agent_name}**")
            for tool_name in tools:
                st.markdown(f"- `{tool_name}`")


def _render_reviewer_dashboard() -> None:
    st.subheader("Reviewer Dashboard")
    st.caption("Claims escalated to human review (project-plan.md Q20/Q72).")

    response = requests.get(f"{BACKEND_URL}/review-queue", headers=_auth_headers(), timeout=10)
    response.raise_for_status()
    items = response.json()

    if not items:
        st.info("No claims currently awaiting review.")
        return

    for item in items:
        with st.container(border=True):
            st.markdown(f"**Claim `{item['claim_ref']}`** — order `{item['order_ref']}`")
            st.markdown(f"Category: {item['claim_category']} · Amount: ${item['refund_amount_usd']:.2f}")
            st.markdown(f"Description: {item['claim_description']}")
            st.markdown(
                f"Image verdict: `{item['image_verdict']}` · Fraud risk: `{item['fraud_risk_band']}`"
            )
            if item["verdict_reasoning"]:
                st.markdown(f"Decision Agent's reasoning: _{item['verdict_reasoning']}_")

            col_approve, col_deny = st.columns(2)
            claim_ref = item["claim_ref"]
            if col_approve.button("Approve (issue refund)", key=f"approve-{claim_ref}"):
                decision = requests.post(
                    f"{BACKEND_URL}/review-queue/{claim_ref}/decide",
                    headers=_auth_headers(),
                    json={"approve": True},
                    timeout=30,
                )
                if decision.status_code == 200:
                    st.success(f"Approved. Transaction: {decision.json().get('transaction_id')}")
                    st.rerun()
                else:
                    st.error(f"Failed: {decision.text}")
            if col_deny.button("Deny", key=f"deny-{claim_ref}"):
                decision = requests.post(
                    f"{BACKEND_URL}/review-queue/{claim_ref}/decide",
                    headers=_auth_headers(),
                    json={"approve": False},
                    timeout=30,
                )
                if decision.status_code == 200:
                    st.success("Denied.")
                    st.rerun()
                else:
                    st.error(f"Failed: {decision.text}")


def main() -> None:
    if not _password_gate():
        return

    st.title("Fraud Triage Admin")
    tab_reviewer, tab_registry = st.tabs(["Reviewer Dashboard", "Tool Registry"])
    with tab_reviewer:
        _render_reviewer_dashboard()
    with tab_registry:
        _render_tool_registry()


main()
