"""LLM-based named entity recognition (testing-and-evaluation-plan.md's
Q9/Q17 -- "Named entity recognition" metric; provider resolved by explicit
user decision to reuse the existing, already-PII-reviewed Azure OpenAI
GPT-4.1 mini deployment rather than adding spaCy or a separate hosted NER
API, mirroring how ragas_report.py reuses this project's own Azure
deployments instead of adding a new provider).

Two real uses per the spec:
1. Entity-level correctness -- do the entities in an agent's free text
   (order ref, product, claim category, monetary amount) match a
   scenario's known ground truth.
2. PII-leak detection -- did a PERSON/GPE/LOC-class entity slip into an
   agent's free text, which fails project-plan.md Q27's "no PII to any
   LLM, under any circumstances" rule.

Same call shape as mcp-servers/orchestrator_server.py's analyze_image:
a real Azure OpenAI chat completion constrained by a JSON-schema
response_format, not a free-text prompt parsed by hand.
"""

import json
import os
from dataclasses import dataclass, field

from openai import AzureOpenAI

_ENTITY_JSON_SCHEMA = {
    "name": "extracted_entities",
    "schema": {
        "type": "object",
        "properties": {
            "persons": {"type": "array", "items": {"type": "string"}},
            "locations": {"type": "array", "items": {"type": "string"}},
            "order_refs": {"type": "array", "items": {"type": "string"}},
            "products": {"type": "array", "items": {"type": "string"}},
            "claim_categories": {"type": "array", "items": {"type": "string"}},
            "monetary_amounts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["persons", "locations", "order_refs", "products", "claim_categories", "monetary_amounts"],
        "additionalProperties": False,
    },
    "strict": True,
}

_EXTRACTION_PROMPT = """Extract named entities from the following text, written by an AI agent \
in an e-commerce returns/refund system. Report:
- persons: any PERSON-class entity -- a human name (customer, reviewer, agent...)
- locations: any GPE/LOC-class entity -- a street address, city, region, or country
- order_refs: any order/claim reference identifiers mentioned
- products: any product name/title mentioned
- claim_categories: any return/refund claim category mentioned (e.g. 'Damaged in Transit')
- monetary_amounts: any dollar amount mentioned (as written, e.g. '$45.00')

Only extract entities genuinely present in the text -- an empty list is correct when a category \
has nothing. Do not infer or guess an entity that isn't explicitly there.

Text:
\"\"\"
{text}
\"\"\""""


@dataclass
class ExtractedEntities:
    persons: list = field(default_factory=list)
    locations: list = field(default_factory=list)
    order_refs: list = field(default_factory=list)
    products: list = field(default_factory=list)
    claim_categories: list = field(default_factory=list)
    monetary_amounts: list = field(default_factory=list)


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def extract_entities(text: str) -> ExtractedEntities:
    if not text or not text.strip():
        return ExtractedEntities()
    response = _client().chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        messages=[{"role": "user", "content": _EXTRACTION_PROMPT.format(text=text)}],
        response_format={"type": "json_schema", "json_schema": _ENTITY_JSON_SCHEMA},
        temperature=0,
    )
    data = json.loads(response.choices[0].message.content)
    return ExtractedEntities(**data)


def embed(texts: list) -> list:
    """text-embedding-3-small (the same deployment ml/rag/embed_and_index.py
    already uses to index the policy chunks), reused rather than adding a
    second embeddings dependency for semantic similarity scoring."""
    response = _client().embeddings.create(model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"], input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
