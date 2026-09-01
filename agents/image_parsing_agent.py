"""Image Parsing Agent. Calls the single, server-side analyze_claim_photo
tool (project-plan.md Q86/Q87) and relays its real, grounded result as the
ConsistencyAssessment -- the agent's job is reporting that result exactly,
not inventing a judgment of its own. analyze_claim_photo (a real Azure
GPT-4.1 mini vision call internally, see mcp-servers/orchestrator_server.py)
is what actually looks at the photo.

This agent never sees or handles the raw photo bytes itself, by design --
project-plan.md Q86/Q87 confirmed a real, 100%-reproducible bug in the
previous 4-tool design (get_photo -> redact_photo -> get_product_reference
-> analyze_image, each a separate real MCP call the agent had to
orchestrate itself): GPT-4.1 mini can't reliably reproduce a ~20K-character
base64 photo blob verbatim as a redact_photo tool-call argument -- it
copies correctly for a while, then silently drifts into a plausible-
looking but fabricated ending. Retries can't fix that kind of long-string
generation drift, since it's systematic, not a one-off glitch -- the real
fix is keeping the blob out of the LLM's own generated arguments entirely,
the same principle this project already applies to every other PII/binary
payload. analyze_claim_photo does the fetch/redact/analyze chain
server-side instead; get_photo/redact_photo/get_product_reference/
analyze_image remain real, independently tested tools in their own right
(mcp-servers/orchestrator_server.py), just no longer part of this agent's
own tool list.
"""

import json
from typing import Any, Optional, Tuple

from crewai import Agent, Task
from crewai.tasks.task_output import TaskOutput

from .llms import get_gpt41_mini_llm
from .schemas import ConsistencyAssessment


def build_image_parsing_agent(tools, customer_ref: Optional[str] = None, claim_ref: Optional[str] = None) -> Agent:
    """customer_ref/claim_ref (Q30/Q73): forwarded to get_gpt41_mini_llm so
    this agent's real LLM calls get tagged for Langfuse tracing."""
    return Agent(
        role="Image Parsing Agent",
        goal=(
            "Judge whether a claim's photo evidence is consistent with the customer's "
            "described issue and matches the ordered product."
        ),
        backstory=(
            "You assess photo evidence for return/refund claims by calling "
            "analyze_claim_photo, which retrieves, redacts, and analyzes the claim's "
            "photo entirely on the server -- you never see or handle the raw photo "
            "data yourself. You report exactly what it returns as your verdict, "
            "product_match, and reasoning -- you do not override or second-guess its "
            "real judgment with your own unsupported opinion."
        ),
        llm=get_gpt41_mini_llm(customer_ref, claim_ref),
        tools=tools,
        verbose=True,
    )


def _real_analyze_claim_photo_result(output: TaskOutput) -> Optional[dict]:
    """The actual analyze_claim_photo result, read from the real tool-call
    message rather than trusting the agent's own report of it -- same
    grounding pattern as decision_agent.py's _real_decision_matrix_result.
    Returns None if it never genuinely succeeded."""
    for message in output.messages:
        if message.get("role") == "tool" and message.get("name") == "analyze_claim_photo":
            content = message.get("content") or ""
            if isinstance(content, str) and '"verdict"' in content and "Error calling tool" not in content:
                try:
                    return json.loads(content)
                except (ValueError, TypeError):
                    continue
    return None


def consistency_guardrail(output: TaskOutput) -> Tuple[bool, Any]:
    """Layer 3: the reported verdict/product_match must match
    analyze_claim_photo's real, genuine result exactly -- catches the agent
    inventing a plausible-sounding judgment instead of reporting the real
    one. A schema alone can't catch this: a fabricated ConsistencyAssessment
    is perfectly well-formed JSON."""
    result = output.pydantic
    if result is None:
        return False, "expected a ConsistencyAssessment, got no structured output"

    real_result = _real_analyze_claim_photo_result(output)
    if real_result is None:
        return False, "analyze_claim_photo never genuinely succeeded -- this looks like a fabricated judgment"
    if result.verdict != real_result["verdict"]:
        return False, (
            f"reported verdict={result.verdict!r} does not match analyze_claim_photo's "
            f"real result {real_result['verdict']!r} -- report the tool's actual verdict, don't override it"
        )
    if result.product_match != real_result["product_match"]:
        return False, (
            f"reported product_match={result.product_match!r} does not match analyze_claim_photo's "
            f"real result {real_result['product_match']!r}"
        )
    return True, result


def build_consistency_task(agent: Agent, claim_ref: str, order_ref: str, claim_category: str, claim_description: str) -> Task:
    return Task(
        description=(
            f"Claim {claim_ref} against order {order_ref}. The customer filed a "
            f"'{claim_category}' claim, describing the issue as: \"{claim_description}\".\n\n"
            "Call analyze_claim_photo with claim_ref, order_ref, claim_category, and "
            "claim_description. Report exactly what it returns as your verdict, "
            "product_match, and reasoning -- do not alter its judgment."
        ),
        expected_output="A ConsistencyAssessment with verdict, product_match, and reasoning.",
        agent=agent,
        output_pydantic=ConsistencyAssessment,
        guardrail=consistency_guardrail,
    )
