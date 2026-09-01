"""Customer-facing chat frontend (project-plan.md Q78, redesigned Q94) --
a second, separate Streamlit app from admin/app.py, on its own port,
replacing the Twilio/WhatsApp channel (Q71/Q76) that was reverted after
Twilio's Sandbox webhook configuration turned out to be gated behind a
paid account tier on this trial account.

No password gate here, unlike admin/app.py -- this is the public,
customer-facing side, open the same way a real customer-support chat
widget would be. Talks to the exact same POST /messages and
POST /messages/photo backend endpoints the (removed) Twilio webhook used,
so the underlying claim-intake pipeline (_process_message,
_get_or_mint_claim_ref) is identical regardless of channel.

Requires backend/main.py (FastAPI) already running -- this app is a thin
UI over its real endpoints, no direct DB/MCP access of its own.

Run with: uv run streamlit run chat/app.py --server.port 8503
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8002")

st.set_page_config(page_title="Fraud Triage Support", page_icon="💬", layout="centered")

# ---------------------------------------------------------------------------
# Visual identity (project-plan.md Q94): a warm, calm "support desk" palette
# distinct from admin/app.py's control-room identity -- this is the public,
# reassurance-first side of the product, so the design leans approachable
# rather than dense. Injected as CSS since Streamlit's own theming
# (.streamlit/config.toml) is shared across every app started from the same
# working directory and can't give chat/admin genuinely different looks.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

      :root {
        --support-ink: #1E2A2E;
        --support-ink-soft: #5B6B70;
        --support-paper: #F6F9F8;
        --support-card: #FFFFFF;
        --support-line: #DCE6E3;
        --support-accent: #1F7A6C;
        --support-accent-soft: #E3F2EE;
        --support-accent-ink: #14544A;
        --support-good: #2E8B57;
        --support-good-soft: #E4F4EA;
        --support-warn: #B5761A;
        --support-warn-soft: #FBF0DF;
        --support-bad: #C1443C;
        --support-bad-soft: #FBE9E7;
      }

      html, body, [class*="st-"], .stMarkdown, .stTextInput input, .stChatInput textarea {
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
      }

      .stApp { background: var(--support-paper); }

      h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--support-ink) !important; }

      /* Masthead */
      .support-masthead {
        display: flex; align-items: center; gap: 0.75rem;
        padding-bottom: 1rem; margin-bottom: 1.25rem;
        border-bottom: 1px solid var(--support-line);
      }
      .support-badge {
        width: 2.6rem; height: 2.6rem; border-radius: 0.85rem;
        background: var(--support-accent); color: white;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.15rem;
        flex-shrink: 0;
      }
      .support-title { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.5rem; color: var(--support-ink); line-height: 1.1; }
      .support-subtitle { color: var(--support-ink-soft); font-size: 0.9rem; margin-top: 0.1rem; }

      /* Chat bubbles */
      [data-testid="stChatMessage"] {
        background: var(--support-card);
        border: 1px solid var(--support-line);
        border-radius: 0.9rem;
        padding: 0.4rem 0.2rem;
      }

      /* Status pill, used inline in assistant replies via markdown */
      .status-pill {
        display: inline-block; font-family: 'Manrope', sans-serif; font-weight: 700;
        font-size: 0.72rem; letter-spacing: 0.03em; text-transform: uppercase;
        padding: 0.18rem 0.55rem; border-radius: 999px; margin-bottom: 0.35rem;
      }
      .status-good { background: var(--support-good-soft); color: var(--support-good); }
      .status-warn { background: var(--support-warn-soft); color: var(--support-warn); }
      .status-bad { background: var(--support-bad-soft); color: var(--support-bad); }
      .status-info { background: var(--support-accent-soft); color: var(--support-accent-ink); }

      /* Buttons */
      .stButton button {
        background: var(--support-accent); color: white; border: none;
        border-radius: 0.6rem; font-family: 'Manrope', sans-serif; font-weight: 700;
      }
      .stButton button:hover { background: var(--support-accent-ink); color: white; }

      /* Sidebar */
      section[data-testid="stSidebar"] { background: var(--support-card); border-right: 1px solid var(--support-line); }
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_sidebar_help() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
              <div class="support-badge" style="width:2.1rem;height:2.1rem;font-size:0.95rem;">?</div>
              <div style="font-family:'Manrope',sans-serif;font-weight:800;font-size:1.05rem;">How this works</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
1. **Sign in** with the email or customer ID on your order — no password needed.
2. **Describe your issue** in plain language: what happened, which order, roughly when. For example:
   *"Order 10 arrived damaged, the box was crushed. It was 2 days ago."*
3. If your claim category needs one, we'll **ask for a photo** — attach it with the uploader that appears below the chat.
4. After that, **send one more short message** (e.g. "here's the photo") so we can finish reviewing your claim — uploading a photo alone doesn't restart the review.
5. You'll get a real verdict: **approved**, **denied**, **escalated to a human reviewer**, or **more information needed**.
            """
        )
        st.divider()
        st.caption("Claims can take a few minutes to process — we're running real checks, not a canned response.")
        st.caption("Limit: 3 new claims per hour per customer. Follow-up messages on an already-open claim don't count against this.")
        if st.session_state.get("customer_identifier"):
            st.divider()
            if st.button("Start a new session"):
                for key in ("customer_identifier", "chat_history", "last_uploaded_photo_name"):
                    st.session_state.pop(key, None)
                st.rerun()


def _identity_gate() -> bool:
    """A raw email/customer ID (Q27) -- resolved to customer_ref entirely
    server-side by every backend call from here on; this page itself never
    sees or stores a customer_ref, only the raw identifier the customer
    typed, exactly the same PII boundary the Twilio channel had."""
    _render_sidebar_help()
    if st.session_state.get("customer_identifier"):
        return True

    st.markdown(
        """
        <div class="support-masthead">
          <div class="support-badge">FT</div>
          <div>
            <div class="support-title">Fraud Triage Support</div>
            <div class="support-subtitle">Report a damaged, wrong, or unwanted item — get a real answer.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("👋 New here? Check **How this works** in the sidebar for a quick walkthrough before you start.")
    identifier = st.text_input("Email or customer ID", placeholder="you@example.com")
    if st.button("Start chat", type="primary") and identifier.strip():
        st.session_state["customer_identifier"] = identifier.strip()
        st.session_state["chat_history"] = []
        st.rerun()
    return False


_OUTCOME_STYLE = {
    "refunded": ("status-good", "✅ Refund issued"),
    "approve": ("status-good", "✅ Approved"),
    "deny": ("status-bad", "❌ Not approved"),
    "escalate": ("status-warn", "🕓 Escalated for human review"),
    "re_prompt_for_photo": ("status-info", "📎 Photo needed"),
    "decision_unavailable": ("status-warn", "🕓 Still processing"),
    "image_analysis_unavailable": ("status-warn", "🕓 Still processing"),
}


def _claim_result_to_chat_text(result: dict) -> str:
    claim_ref = result["claim_ref"]
    outcome = result["outcome"]
    css_class, label = _OUTCOME_STYLE.get(outcome, ("status-info", outcome))
    pill = f'<span class="status-pill {css_class}">{label}</span><br>'

    if outcome == "re_prompt_for_photo":
        body = f"Claim `{claim_ref}`: this category needs a photo of the item/issue. Please attach one using the uploader below, then send one more message to continue."
    elif outcome == "refunded":
        body = f"Claim `{claim_ref}` was approved and your refund has been issued (transaction `{result.get('transaction_id')}`)."
    elif outcome == "deny":
        body = f"Claim `{claim_ref}` was reviewed and could not be approved under our current policy."
    elif outcome == "escalate":
        body = f"Claim `{claim_ref}` needs a closer look from our review team — we'll follow up soon."
    elif outcome in ("decision_unavailable", "image_analysis_unavailable"):
        body = f"Claim `{claim_ref}` is still being processed — we'll follow up shortly."
    else:
        body = f"Claim `{claim_ref}`: {outcome}."
    return pill + body


def _message_response_to_chat_text(response: dict) -> str:
    if response.get("needs_more_info"):
        return response.get("follow_up_question") or "Could you share a few more details about your claim?"
    if response.get("claim_result"):
        return _claim_result_to_chat_text(response["claim_result"])
    if response.get("request_type") == "general_inquiry":
        # A real, honest gap, not papered over: the Orchestrator's intake
        # classification doesn't generate a policy-question answer (Q67's
        # ClaimIntakeResult only ever extracts claim fields or asks a
        # follow-up) -- a specific "your policy question" answering flow
        # wasn't asked for in this pass.
        return "That looks like a general question rather than a specific order issue. Please describe the order/item you'd like help with, and I can open a claim."
    return "Thanks for your message — we're looking into it."


def _send_message(text: str) -> None:
    response = requests.post(
        f"{BACKEND_URL}/messages",
        json={"customer_identifier": st.session_state["customer_identifier"], "message": text},
        # project-plan.md Q90/Q91: 180s was confirmed too short for real
        # traffic -- a claim that already has its photo chains Image
        # Parsing (real vision call) -> Fraud Scoring (real DynamoDB/
        # Postgres + SHAP-explained ML scoring) -> Decision (real
        # OpenSearch retrieval + LLM call) in one request, and any
        # guardrail retry at any of those stages adds a full extra round
        # trip. Real backend logs showed claims genuinely completing past
        # 180s, not hanging -- 480s gives real headroom without masking
        # an actual hang (the Flow itself has no server-side timeout, so
        # a truly stuck request would still eventually surface here).
        timeout=480,
    )
    if response.status_code == 429:
        reply = '<span class="status-pill status-warn">⏳ Rate limited</span><br>You\'ve hit the claim-submission rate limit for now — please try again later.'
    elif response.status_code != 200:
        reply = f'<span class="status-pill status-bad">⚠️ Error</span><br>Something went wrong (HTTP {response.status_code}). Please try again.'
    else:
        reply = _message_response_to_chat_text(response.json())
    st.session_state["chat_history"].append({"role": "assistant", "content": reply})


def _send_photo(uploaded_file) -> None:
    files = {"photo": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    data = {"customer_identifier": st.session_state["customer_identifier"]}
    response = requests.post(f"{BACKEND_URL}/messages/photo", data=data, files=files, timeout=30)
    if response.status_code == 200:
        claim_ref = response.json()["claim_ref"]
        reply = (
            f'<span class="status-pill status-info">📎 Photo received</span><br>'
            f"Photo received for claim `{claim_ref}`. Please send one more message to continue your claim."
        )
    else:
        reply = f'<span class="status-pill status-bad">⚠️ Upload failed</span><br>Photo upload failed (HTTP {response.status_code}). Please try again.'
    st.session_state["chat_history"].append({"role": "assistant", "content": reply})


def main() -> None:
    if not _identity_gate():
        return

    st.markdown(
        """
        <div class="support-masthead">
          <div class="support-badge">FT</div>
          <div>
            <div class="support-title">Fraud Triage Support</div>
            <div class="support-subtitle">Signed in as your email/customer ID — never shared with any AI model.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state["chat_history"]:
        st.markdown(
            '<span class="status-pill status-info">👋 Getting started</span><br>'
            "Tell me what happened — the order, what went wrong, and roughly when. "
            "I'll ask for a photo if your claim category needs one.",
            unsafe_allow_html=True,
        )

    for message in st.session_state["chat_history"]:
        avatar = "🧑" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📎 Attach a photo (only if asked — for damaged/incorrect item claims)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None and uploaded_file.name != st.session_state.get("last_uploaded_photo_name"):
        st.session_state["last_uploaded_photo_name"] = uploaded_file.name
        st.session_state["chat_history"].append({"role": "user", "content": f"📎 Attached photo: {uploaded_file.name}"})
        with st.spinner("Uploading photo..."):
            _send_photo(uploaded_file)
        st.rerun()

    user_text = st.chat_input("Describe your order issue, or ask a question...")
    if user_text:
        st.session_state["chat_history"].append({"role": "user", "content": user_text})
        with st.spinner("Reviewing your claim — this can take a few minutes for a full review..."):
            _send_message(user_text)
        st.rerun()


main()
