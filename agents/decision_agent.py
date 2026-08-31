"""Decision Agent. Weighs the Image Parsing Agent's photo-consistency
verdict and the Fraud Scoring Agent's risk assessment against the real
refund policy to reach a Verdict -- the only agent whose output can ever
trigger issue_refund (and even then, only via Flow-level code checking
Verdict.decision == "approve"; this agent never calls issue_refund itself).

The approve/deny/escalate decision is grounded in apply_decision_matrix (a
real MCP tool, not LLM judgment -- see its docstring in
mcp-servers/orchestrator_server.py for why). This agent's real
contribution is retrieving relevant policy text via search_refund_policy
and determining refund_form/policy_clause/reasoning from what it actually
returns.

Runs on GPT-4.1 mini via Azure OpenAI (Q52/Q75) -- originally speced as a
local Ollama + DeepSeek-R1-Distill-Qwen-8B model kept local by explicit
choice, but that never produced a genuine, non-fabricated verdict in
practice (Q70) and was dropped in favor of this project's other three
agents' model, removing the local-inference dependency entirely.
"""

from typing import Any, Optional, Tuple

from crewai import Agent, Task
from crewai.tasks.task_output import TaskOutput

from .llms import get_decision_llm
from .schemas import Verdict


def build_decision_agent(tools, customer_ref: Optional[str] = None, claim_ref: Optional[str] = None) -> Agent:
    """customer_ref/claim_ref (Q30/Q73): forwarded to get_decision_llm so
    this agent's real LLM calls get tagged for Langfuse tracing."""
    return Agent(
        role="Decision Agent",
        goal=(
            "Weigh the Image Parsing Agent's photo-consistency verdict and the Fraud Scoring "
            "Agent's risk assessment against the real refund policy to reach a final claim decision."
        ),
        backstory=(
            "You are the final, non-bypassable gate on every return/refund claim. The "
            "approve/deny/escalate decision itself is not yours to invent -- you call "
            "apply_decision_matrix with the real image-consistency verdict, fraud risk band, "
            "and refund amount, and report exactly what it returns, never a different label. "
            "Your own judgment goes into retrieving the relevant refund-policy clause via "
            "search_refund_policy and determining refund_form and policy_clause/policy_version "
            "from what it actually returns -- and writing reasoning that reflects the real "
            "inputs, not a plausible-sounding invention. If search_refund_policy comes back "
            "not confident (no relevant chunk found), you escalate regardless of what "
            "apply_decision_matrix would otherwise say."
        ),
        llm=get_decision_llm(customer_ref, claim_ref),
        tools=tools,
        verbose=True,
    )


def _tool_genuinely_succeeded(output: TaskOutput, tool_name: str, success_marker: str) -> bool:
    """Same grounding check used by Image Parsing's and Fraud Scoring's
    guardrails: scans the task's real message history for a tool result
    that actually looks like success, not just "was this tool called"."""
    for message in output.messages:
        if message.get("role") == "tool" and message.get("name") == tool_name:
            content = message.get("content") or ""
            if isinstance(content, str) and success_marker in content and "Error calling tool" not in content:
                return True
    return False


def _real_decision_matrix_result(output: TaskOutput) -> str | None:
    """The actual decision apply_decision_matrix returned, read from the
    real tool-call message rather than trusting the agent's own report of
    it -- returns None if it never genuinely succeeded."""
    for message in output.messages:
        if message.get("role") == "tool" and message.get("name") == "apply_decision_matrix":
            content = message.get("content") or ""
            if isinstance(content, str) and '"decision"' in content and "Error calling tool" not in content:
                import json  # noqa: PLC0415

                try:
                    return json.loads(content).get("decision")
                except (ValueError, TypeError):
                    continue
    return None


def verdict_guardrail(output: TaskOutput) -> Tuple[bool, Any]:
    """Layer 3: the decision must match apply_decision_matrix's real,
    genuine result exactly -- catches an agent reporting a plausible but
    invented decision, the same fabrication pattern proven twice in
    Image Parsing's guardrail (see its docstring). Also requires
    search_refund_policy to have genuinely succeeded, since policy_clause/
    policy_version/refund_form should trace back to real retrieved text,
    not the agent's own guess at what the policy says."""
    result = output.pydantic
    if result is None:
        return False, "expected a Verdict, got no structured output"

    real_decision = _real_decision_matrix_result(output)
    if real_decision is None:
        return False, "apply_decision_matrix never genuinely succeeded -- this looks like a fabricated verdict"
    if result.decision != real_decision:
        return False, (
            f"reported decision={result.decision!r} does not match apply_decision_matrix's "
            f"real result {real_decision!r} -- report the tool's actual decision, don't override it"
        )
    if not _tool_genuinely_succeeded(output, "search_refund_policy", '"confident"'):
        return False, "search_refund_policy never genuinely succeeded -- policy_clause/refund_form must be grounded in real retrieval"
    return True, result


def build_verdict_task(
    agent: Agent,
    claim_ref: str,
    order_ref: str,
    claim_category: str,
    claim_description: str,
    refund_amount_usd: float,
    image_verdict: str,
    fraud_risk_band: str,
    fraud_key_signals: list,
) -> Task:
    return Task(
        description=(
            f"Reach a final verdict on claim {claim_ref} (order {order_ref}), category "
            f"'{claim_category}', described as: \"{claim_description}\". Refund amount claimed: "
            f"${refund_amount_usd:.2f}. The Image Parsing Agent's consistency verdict was "
            f"'{image_verdict}'. The Fraud Scoring Agent's risk band was '{fraud_risk_band}', "
            f"driven by these key signals: {fraud_key_signals}.\n\n"
            "1. Call search_refund_policy with a query describing this claim's category and "
            "situation, to retrieve the relevant policy clause(s). If it comes back not "
            "confident, you must escalate regardless of the next step's result.\n"
            f"2. Call apply_decision_matrix with image_verdict='{image_verdict}', "
            f"fraud_risk_band='{fraud_risk_band}', refund_amount_usd={refund_amount_usd}. "
            "Report its exact decision -- never a different label, even if you'd reason "
            "differently yourself.\n"
            "3. Based on the retrieved policy text, determine refund_form ('original_payment_"
            "method' or 'store_credit') and cite the specific policy_clause and policy_version "
            "from what search_refund_policy actually returned.\n"
            "4. Write reasoning that explains the real decision using the real signals above -- "
            "never invent a justification unconnected to the actual tool results."
        ),
        expected_output="A Verdict with decision, refund_amount, refund_form, policy_clause, policy_version, and reasoning.",
        agent=agent,
        output_pydantic=Verdict,
        guardrail=verdict_guardrail,
    )
