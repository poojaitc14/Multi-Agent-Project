"""Chunk docs/refund_policy.md by clause/category (project-plan.md Q56).

One chunk per claim category, one for the decision matrix, one for the
guardrail -- plus the standalone "Return window" rule, since it's a real
rule worth being independently retrievable (e.g. a query about how many
days a customer has to return an item shouldn't need to match a specific
category first). No fixed-size/token-window splitting.
"""

import re
from pathlib import Path

POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "refund_policy.md"


def parse_version(text: str) -> str:
    match = re.match(r"Version:\s*(\S+)", text.strip())
    if not match:
        raise ValueError("docs/refund_policy.md must start with a 'Version: vX.Y' header line")
    return match.group(1)


def chunk_policy(text: str) -> list[dict]:
    """Splits on level-2 (##) and level-3 (###) headers. Each claim category
    is its own ### under "## Claim categories" and becomes its own chunk;
    "Return window", "Decision matrix", and "Guardrail" are top-level ##
    sections and each become their own chunk too."""
    version = parse_version(text)
    body = text.split("\n", 1)[1] if "\n" in text else ""

    # Split on any header line (## or ###), keeping the header with its content
    sections = re.split(r"\n(?=#{2,3} )", body.strip())

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        header_match = re.match(r"^(#{2,3})\s+(.+)", section)
        if not header_match:
            continue
        level, title = header_match.groups()
        if title.strip() == "Claim categories":
            # This header itself has no content of its own -- its ### children
            # are separate sections in the split above and become their own chunks.
            continue
        chunks.append(
            {
                "chunk_id": title.strip().lower().replace(" ", "_").replace("/", "_"),
                "title": title.strip(),
                "text": section,
                "policy_version": version,
            }
        )
    return chunks


if __name__ == "__main__":
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    result = chunk_policy(policy_text)
    print(f"Parsed {len(result)} chunks from policy {result[0]['policy_version'] if result else '?'}:")
    for c in result:
        print(f"  - {c['chunk_id']!r} ({len(c['text'])} chars)")
