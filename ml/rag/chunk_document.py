"""Generic chunking for admin-uploaded documents (project-plan.md Q96),
separate from chunk_policy.py -- that one is deliberately special-cased to
docs/refund_policy.md's exact "## Claim categories" / "### <category>"
structure and a required "Version: vX.Y" first line; an arbitrary uploaded
document has neither, so it needs its own, structure-agnostic splitter
rather than being forced through chunk_policy's parser.

Paragraph-based, not fixed-size/token-window: splits on blank lines, then
greedily packs consecutive paragraphs into chunks up to MAX_CHUNK_CHARS so
a chunk never cuts a paragraph in half. A single paragraph longer than the
limit still becomes its own (oversized) chunk rather than being silently
truncated -- an accepted tradeoff for a first version of admin-uploaded
document support.
"""

import re
import uuid

MAX_CHUNK_CHARS = 1200


def chunk_document(text: str, source_document: str) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[dict] = []
    current: list[str] = []
    current_len = 0

    def _flush() -> None:
        if not current:
            return
        index = len(chunks)
        body = "\n\n".join(current)
        chunks.append(
            {
                "chunk_id": f"{source_document}::{index}::{uuid.uuid4().hex[:8]}",
                "title": f"{source_document} (part {index + 1})",
                "text": body,
                "policy_version": "uploaded",
                "source_document": source_document,
            }
        )

    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 2 > MAX_CHUNK_CHARS:
            _flush()
            current, current_len = [], 0
        current.append(paragraph)
        current_len += len(paragraph) + 2

    _flush()
    return chunks
