"""Admin app (project-plan.md Q71/Q78, redesigned Q94): the internal-only
Streamlit scope, separate from chat/app.py's customer-facing chat
frontend. Two sections, both behind the same shared reviewer password
(Q31):

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

st.set_page_config(page_title="Fraud Triage Admin", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# Visual identity (project-plan.md Q94): a deliberately different, more
# structured "control room" palette from chat/app.py's warm support-desk
# identity -- this is the internal, data-dense side of the product.
# Injected as CSS for the same reason chat/app.py does: Streamlit's own
# theming is shared across every app started from the same working
# directory and can't give each app a genuinely different look.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

      :root {
        --admin-ink: #171C24;
        --admin-ink-soft: #565F70;
        --admin-paper: #F2F4F8;
        --admin-card: #FFFFFF;
        --admin-line: #DCE1EB;
        --admin-accent: #33436B;
        --admin-accent-soft: #E4E9F3;
        --admin-accent-ink: #202D4A;
        --admin-good: #227A54;
        --admin-good-soft: #E1F1E9;
        --admin-warn: #A65C13;
        --admin-warn-soft: #FBEBD8;
        --admin-bad: #B33A34;
        --admin-bad-soft: #F9E4E2;
      }

      html, body, [class*="st-"], .stMarkdown, .stTextInput input {
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
      }

      .stApp { background: var(--admin-paper); }
      h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; color: var(--admin-ink) !important; }
      code, .admin-mono { font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }

      .admin-masthead {
        display: flex; align-items: center; gap: 0.85rem;
        padding-bottom: 1rem; margin-bottom: 1.25rem;
        border-bottom: 2px solid var(--admin-line);
      }
      .admin-badge {
        width: 2.6rem; height: 2.6rem; border-radius: 0.5rem;
        background: var(--admin-accent); color: white;
        display: flex; align-items: center; justify-content: center;
        font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 1rem;
        flex-shrink: 0;
      }
      .admin-title { font-family: 'IBM Plex Sans', sans-serif; font-weight: 700; font-size: 1.5rem; color: var(--admin-ink); line-height: 1.1; }
      .admin-subtitle { color: var(--admin-ink-soft); font-size: 0.85rem; margin-top: 0.1rem; font-family: 'IBM Plex Mono', monospace; }

      /* Status pill for claim verdicts */
      .status-pill {
        display: inline-block; font-family: 'IBM Plex Mono', monospace; font-weight: 600;
        font-size: 0.72rem; letter-spacing: 0.03em; text-transform: uppercase;
        padding: 0.2rem 0.6rem; border-radius: 3px;
      }
      .status-good { background: var(--admin-good-soft); color: var(--admin-good); }
      .status-warn { background: var(--admin-warn-soft); color: var(--admin-warn); }
      .status-bad { background: var(--admin-bad-soft); color: var(--admin-bad); }
      .status-info { background: var(--admin-accent-soft); color: var(--admin-accent-ink); }

      /* Claim cards */
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--admin-card); border: 1px solid var(--admin-line) !important;
        border-radius: 0.5rem;
      }

      .stButton button { border-radius: 0.4rem; font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; }
      button[kind="primary"] { background: var(--admin-good) !important; border: none !important; }
      section[data-testid="stSidebar"] { background: var(--admin-card); border-right: 1px solid var(--admin-line); }
    </style>
    """,
    unsafe_allow_html=True,
)


def _auth_headers() -> dict:
    return {"X-Reviewer-Password": st.session_state.get("reviewer_password", "")}


def _render_sidebar_help() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
              <div class="admin-badge" style="width:2rem;height:2rem;font-size:0.8rem;">i</div>
              <div style="font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:1rem;">Operating guide</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
**Reviewer Dashboard**
1. Every claim here has outcome `escalate` — the Decision Agent's own guardrail already forced human review (e.g. amount over $200, or high fraud risk / inconsistent photo evidence).
2. Read the claim's category, amount, description, and the real image/fraud verdicts before deciding.
3. **Approve** issues a real refund through the same gated `issue_refund` path the system itself uses — this is not reversible.
4. **Deny** closes the claim with no refund.

**Tool Registry**
Read-only. Shows exactly which MCP tools each agent is wired to call, straight from the real agent code — not a document that can drift out of date.
            """
        )
        st.divider()
        st.caption("All actions here are real — approvals move real money through the real payment path.")
        if st.session_state.get("reviewer_authenticated"):
            st.divider()
            if st.button("Sign out"):
                for key in ("reviewer_authenticated", "reviewer_password"):
                    st.session_state.pop(key, None)
                st.rerun()


def _password_gate() -> bool:
    """Q31: nothing below this renders until the shared reviewer password
    is entered -- checked against the real backend (GET /tool-registry is
    the cheapest real endpoint behind require_reviewer), not just compared
    to a value this page happens to know, so a wrong/stale password fails
    the same way it would against any other protected endpoint."""
    _render_sidebar_help()
    if st.session_state.get("reviewer_authenticated"):
        return True

    st.markdown(
        """
        <div class="admin-masthead">
          <div class="admin-badge">FT</div>
          <div>
            <div class="admin-title">Fraud Triage Admin</div>
            <div class="admin-subtitle">reviewer console · internal only</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("🔒 Reviewer access only. See **Operating guide** in the sidebar before approving or denying a claim.")
    password = st.text_input("Reviewer password", type="password")
    if st.button("Sign in", type="primary") and password:
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
    st.caption("Read-only — which real MCP tools each agent is wired to call.")
    response = requests.get(f"{BACKEND_URL}/tool-registry", headers=_auth_headers(), timeout=10)
    response.raise_for_status()
    registry = response.json()
    cols = st.columns(min(len(registry), 4)) if registry else []
    for (agent_name, tools), col in zip(registry.items(), cols):
        with col:
            with st.container(border=True):
                st.markdown(f"**{agent_name}**")
                for tool_name in tools:
                    st.markdown(f'<span class="admin-mono" style="font-size:0.85rem;">{tool_name}</span>', unsafe_allow_html=True)


_RISK_STYLE = {"low": "status-good", "medium": "status-warn", "high": "status-bad"}
_VERDICT_STYLE = {"consistent": "status-good", "partially_consistent": "status-warn", "inconsistent": "status-bad", "no_photo": "status-info"}


def _pill(text: str, css_class: str) -> str:
    return f'<span class="status-pill {css_class}">{text}</span>'


def _render_reviewer_dashboard() -> None:
    st.subheader("Reviewer Dashboard")
    st.caption("Claims escalated to human review (project-plan.md Q20/Q72).")

    response = requests.get(f"{BACKEND_URL}/review-queue", headers=_auth_headers(), timeout=10)
    response.raise_for_status()
    items = response.json()

    if not items:
        st.success("✅ Queue is empty — no claims currently awaiting review.")
        return

    st.markdown(f'<span class="status-pill status-warn">⏳ {len(items)} awaiting review</span>', unsafe_allow_html=True)
    st.write("")

    for item in items:
        with st.container(border=True):
            header_col, amount_col = st.columns([3, 1])
            with header_col:
                st.markdown(
                    f'<span class="admin-mono" style="font-weight:600;">clm/{item["claim_ref"]}</span> '
                    f'&nbsp;·&nbsp; order <span class="admin-mono">{item["order_ref"]}</span> '
                    f'&nbsp;·&nbsp; {item["claim_category"]}',
                    unsafe_allow_html=True,
                )
            with amount_col:
                st.markdown(
                    f'<div style="text-align:right;font-family:\'IBM Plex Mono\',monospace;font-weight:600;font-size:1.05rem;">${item["refund_amount_usd"]:.2f}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(f'_{item["claim_description"]}_')

            image_class = _VERDICT_STYLE.get(item["image_verdict"], "status-info")
            fraud_class = _RISK_STYLE.get(item["fraud_risk_band"], "status-info")
            st.markdown(
                _pill(f'image: {item["image_verdict"]}', image_class) + "&nbsp;&nbsp;" +
                _pill(f'fraud risk: {item["fraud_risk_band"]}', fraud_class),
                unsafe_allow_html=True,
            )

            if item["verdict_reasoning"]:
                with st.expander("Decision Agent's reasoning"):
                    st.write(item["verdict_reasoning"])

            col_approve, col_deny = st.columns(2)
            claim_ref = item["claim_ref"]
            if col_approve.button("✅ Approve (issue refund)", key=f"approve-{claim_ref}", type="primary"):
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
            if col_deny.button("❌ Deny", key=f"deny-{claim_ref}"):
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

    st.markdown(
        """
        <div class="admin-masthead">
          <div class="admin-badge">FT</div>
          <div>
            <div class="admin-title">Fraud Triage Admin</div>
            <div class="admin-subtitle">reviewer console · internal only</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_reviewer, tab_registry = st.tabs(["📋 Reviewer Dashboard", "🔧 Tool Registry"])
    with tab_reviewer:
        _render_reviewer_dashboard()
    with tab_registry:
        _render_tool_registry()


main()
