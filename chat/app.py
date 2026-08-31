"""Customer-facing chat frontend (project-plan.md Q78) -- a second,
separate Streamlit app from admin/app.py, on its own port, replacing the
Twilio/WhatsApp channel (Q71/Q76) that was reverted after Twilio's Sandbox
webhook configuration turned out to be gated behind a paid account tier on
this trial account.

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

st.set_page_config(page_title="Fraud Triage Support Chat", layout="centered")


def _identity_gate() -> bool:
    """A raw email/customer ID (Q27) -- resolved to customer_ref entirely
    server-side by every backend call from here on; this page itself never
    sees or stores a customer_ref, only the raw identifier the customer
    typed, exactly the same PII boundary the Twilio channel had."""
    if st.session_state.get("customer_identifier"):
        return True

    st.title("Fraud Triage Support")
    st.write("Enter your email or customer ID to start a claim or ask a question.")
    identifier = st.text_input("Email or customer ID")
    if st.button("Start chat") and identifier.strip():
        st.session_state["customer_identifier"] = identifier.strip()
        st.session_state["chat_history"] = []
        st.rerun()
    return False


def _claim_result_to_chat_text(result: dict) -> str:
    claim_ref = result["claim_ref"]
    outcome = result["outcome"]
    if outcome == "re_prompt_for_photo":
        return f"Claim `{claim_ref}`: this category needs a photo of the item/issue. Please attach one using the uploader below."
    if outcome == "refunded":
        return f"Good news — claim `{claim_ref}` was approved and your refund has been issued (transaction `{result.get('transaction_id')}`)."
    if outcome == "deny":
        return f"Claim `{claim_ref}` was reviewed and could not be approved under our current policy."
    if outcome == "escalate":
        return f"Claim `{claim_ref}` needs a closer look from our review team — we'll follow up soon."
    if outcome in ("decision_unavailable", "image_analysis_unavailable"):
        return f"Claim `{claim_ref}` is still being processed — we'll follow up shortly."
    return f"Claim `{claim_ref}`: {outcome}."


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
        timeout=180,  # the real CrewAI Flow behind this can take a while -- multiple real LLM calls
    )
    if response.status_code == 429:
        reply = "You've hit the claim-submission rate limit for now — please try again later."
    elif response.status_code != 200:
        reply = f"Something went wrong (HTTP {response.status_code}). Please try again."
    else:
        reply = _message_response_to_chat_text(response.json())
    st.session_state["chat_history"].append({"role": "assistant", "content": reply})


def _send_photo(uploaded_file) -> None:
    files = {"photo": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    data = {"customer_identifier": st.session_state["customer_identifier"]}
    response = requests.post(f"{BACKEND_URL}/messages/photo", data=data, files=files, timeout=30)
    if response.status_code == 200:
        claim_ref = response.json()["claim_ref"]
        reply = f"Photo received for claim `{claim_ref}`. Please continue describing your issue in the chat if you haven't already."
    else:
        reply = f"Photo upload failed (HTTP {response.status_code}). Please try again."
    st.session_state["chat_history"].append({"role": "assistant", "content": reply})


def main() -> None:
    if not _identity_gate():
        return

    st.title("Fraud Triage Support")

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    uploaded_file = st.file_uploader("Attach a photo (optional, for damaged/incorrect item claims)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None and uploaded_file.name != st.session_state.get("last_uploaded_photo_name"):
        st.session_state["last_uploaded_photo_name"] = uploaded_file.name
        st.session_state["chat_history"].append({"role": "user", "content": f"📎 Attached photo: {uploaded_file.name}"})
        with st.spinner("Uploading photo..."):
            _send_photo(uploaded_file)
        st.rerun()

    user_text = st.chat_input("Describe your order issue, or ask a question...")
    if user_text:
        st.session_state["chat_history"].append({"role": "user", "content": user_text})
        with st.spinner("Thinking..."):
            _send_message(user_text)
        st.rerun()


main()
