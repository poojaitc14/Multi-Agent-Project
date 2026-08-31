"""Image Parsing Agent. Its Task orchestrates get_photo -> redact_photo ->
get_product_reference -> analyze_image and relays analyze_image's real,
grounded verdict as the ConsistencyAssessment -- the agent's job is tool
orchestration, not inventing a judgment of its own; analyze_image (a real
Azure GPT-4.1 mini vision call, see mcp-servers/orchestrator_server.py) is
what actually looks at the photo.

The "no_photo" verdict (ConsistencyAssessment's 4th enum value, not part of
analyze_image's own output -- see its docstring) is decided by the Task
itself: if get_photo finds nothing for the claim, the agent must return
verdict="no_photo" without ever calling analyze_image, since there is
nothing to analyze.
"""

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
            "You assess photo evidence for return/refund claims. If a claim has no "
            "photo, you report verdict='no_photo' and product_match=false without "
            "guessing -- you never fabricate a judgment about a photo that doesn't "
            "exist. When a photo does exist, you always redact it (auto-blur "
            "incidental faces/text) before analyzing it, and you report exactly what "
            "the vision analysis found -- you do not override or second-guess a real "
            "vision judgment with your own unsupported opinion."
        ),
        llm=get_gpt41_mini_llm(customer_ref, claim_ref),
        tools=tools,
        verbose=True,
    )


_PHOTO_TOOLS = {"get_photo", "redact_photo", "analyze_image"}


def _analyze_image_genuinely_succeeded(output: TaskOutput) -> bool:
    """Real, repeatedly observed failure mode (not hypothetical): base64
    photo data has to pass through the agent's own tool-call argument
    generation at TWO hops -- get_photo's output retyped as redact_photo's
    input, then redact_photo's output retyped as analyze_image's input --
    and an LLM can corrupt a long base64 blob at either hop. When that
    happens, CrewAI catches the resulting exception and writes an
    "Error calling tool ..." string into the tool-role message's content
    rather than raising, so a naive check for "did analyze_image run at
    all" isn't enough -- it can have run and still failed. This scans for
    a tool-role 'analyze_image' message whose content is genuinely the
    expected JSON shape (a 'verdict' key), not an error string standing in
    its place."""
    for message in output.messages:
        if message.get("role") == "tool" and message.get("name") == "analyze_image":
            content = message.get("content") or ""
            if isinstance(content, str) and '"verdict"' in content and "Error calling tool" not in content:
                return True
    return False


def _get_photo_genuinely_succeeded(output: TaskOutput) -> bool:
    """The other direction of the same fabrication pattern, caught by a
    real end-to-end Flow run (Slice 9): a photo genuinely existed and
    get_photo genuinely returned it, but redact_photo or analyze_image
    then failed on the retyped base64 -- and the agent's cheapest way out
    was reporting verdict='no_photo', which the earlier version of this
    guardrail didn't catch (it only checked the verdict!='no_photo' + tool-
    failure direction, not this one). get_photo returns raw base64 on
    success (no JSON wrapper, unlike analyze_image), so success here is
    just 'a message exists and isn't an error string'."""
    for message in output.messages:
        if message.get("role") == "tool" and message.get("name") == "get_photo":
            content = message.get("content") or ""
            if isinstance(content, str) and content and "Error calling tool" not in content:
                return True
    return False


def consistency_guardrail(output: TaskOutput) -> Tuple[bool, Any]:
    """Layer 3: catches a logically inconsistent combination a schema alone
    allows (verdict='no_photo' with product_match=true is nonsensical --
    there is no photo to match), and -- the more important checks, added
    after reproducing real fabrication three separate times while building
    this -- catches the agent inventing a plausible-sounding verdict/
    reasoning in EITHER direction: claiming a photo was analyzed when
    analyze_image never genuinely succeeded, or claiming no_photo when
    get_photo actually found one and a later step just failed. A schema
    alone can't catch either: a fabricated ConsistencyAssessment is
    perfectly well-formed JSON."""
    result = output.pydantic
    if result is None:
        return False, "expected a ConsistencyAssessment, got no structured output"
    if result.verdict == "no_photo" and result.product_match:
        return False, "verdict='no_photo' cannot have product_match=true -- there is no photo to match"
    if result.verdict == "no_photo" and _get_photo_genuinely_succeeded(output):
        return False, (
            "verdict='no_photo' but get_photo actually found one -- a later step "
            "(redact_photo/analyze_image) failed instead; retry rather than reporting no_photo"
        )
    if result.verdict != "no_photo" and not _analyze_image_genuinely_succeeded(output):
        return False, (
            f"verdict={result.verdict!r} was reported but analyze_image never genuinely "
            "succeeded (never called, or called and errored) -- this looks like a fabricated "
            "judgment, not a real one; retry get_photo/redact_photo/analyze_image"
        )
    return True, result


def build_consistency_task(agent: Agent, claim_ref: str, order_ref: str, claim_category: str, claim_description: str) -> Task:
    return Task(
        description=(
            f"Claim {claim_ref} against order {order_ref}. The customer filed a "
            f"'{claim_category}' claim, describing the issue as: \"{claim_description}\".\n\n"
            "1. Call get_photo to retrieve the claim's photo. If none exists, return "
            "verdict='no_photo', product_match=false, and explain that in reasoning -- "
            "do not call analyze_image.\n"
            "2. If a photo exists, call redact_photo on it first -- never analyze an "
            "unredacted photo.\n"
            "3. Call get_product_reference for the ordered product's title/description.\n"
            "4. Call analyze_image with the redacted photo, the claim category/"
            "description, and the product reference. Report exactly what it returns "
            "as your verdict, product_match, and reasoning -- do not alter its judgment."
        ),
        expected_output="A ConsistencyAssessment with verdict, product_match, and reasoning.",
        agent=agent,
        output_pydantic=ConsistencyAssessment,
        guardrail=consistency_guardrail,
    )
