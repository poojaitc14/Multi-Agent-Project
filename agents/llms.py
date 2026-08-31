"""LLM configuration for the 4 agents (project-plan.md's "Agents & models"
table). Reuses the same AZURE_OPENAI_* env vars already used directly by
ml/rag/embed_and_index.py and mcp-servers/orchestrator_server.py's
analyze_image -- crewai.LLM's real fields (verified against the installed
package, not assumed from docs) are api_key/api_base/api_version, not
"endpoint" as crewai's own docs page currently states.
"""

import os
from typing import Optional

from crewai import LLM

from .observability import configure_langfuse, trace_metadata


def get_gpt41_mini_llm(customer_ref: Optional[str] = None, claim_ref: Optional[str] = None) -> LLM:
    """Orchestrator, Fraud Scoring, and Image Parsing's default (Q47/Q52) --
    Azure-hosted GPT-4.1 mini via litellm's azure/ provider prefix.

    is_litellm=True is required, not optional: crewai's LLM otherwise routes
    "azure/..." to its own *native* Azure provider (the azure-ai-inference
    SDK), which targets Azure AI Foundry-style endpoints and 404s against a
    classic Azure OpenAI Cognitive Services resource -- confirmed empirically
    against this project's real resource before adding this flag. litellm's
    Azure integration is what api_base/api_version/api_key are meant for,
    and matches how ml/rag/embed_and_index.py and orchestrator_server.py's
    analyze_image already call this same resource successfully.

    customer_ref/claim_ref (Q30/Q73): when given, every call this LLM makes
    is tagged for Langfuse tracing via additional_params -- customer_ref
    only (claim_ref not yet assigned, e.g. during multi-turn intake) is
    valid; omitting both just means this LLM's calls aren't traced with a
    user_id/session_id (not an error -- Langfuse itself is optional)."""
    configure_langfuse()
    additional_params = trace_metadata(customer_ref, claim_ref) if customer_ref else {}
    return LLM(
        model=f"azure/{os.environ['AZURE_OPENAI_DEPLOYMENT_NAME']}",
        is_litellm=True,
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        additional_params=additional_params,
    )


def get_decision_llm(customer_ref: Optional[str] = None, claim_ref: Optional[str] = None) -> LLM:
    """Decision Agent's model (Q52, revised by Q75): GPT-4.1 mini via Azure
    OpenAI, same as the other three agents -- delegates straight to
    get_gpt41_mini_llm() rather than duplicating its LLM(...) construction.

    Originally DeepSeek-R1-Distill-Qwen-8B via local Ollama, kept local by
    explicit choice (Q52). Dropped by explicit user decision (Q75) after
    CPU-only local inference at the 8B reasoning-model tier proved too slow
    and unreliable to verify in practice (Q70: real runs took 15-45+
    minutes, and the one run that didn't time out still fabricated a
    verdict that verdict_guardrail correctly caught and could never get
    past in 3 retries) -- no genuine, non-fabricated Decision Agent verdict
    was ever obtained via Ollama. GPT-4.1 mini removes both problems at
    once and drops the local-Ollama dependency entirely."""
    return get_gpt41_mini_llm(customer_ref, claim_ref)
