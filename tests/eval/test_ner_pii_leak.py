"""NER-based PII-leak detection (testing-and-evaluation-plan.md's Q9 --
"Named entity recognition", use 2: "run NER over every piece of free text
an agent produces... to catch a person name, street address, or other
identifiable entity that slipped into unstructured text. This complements
the 'PII isolation tests'... those scan structured tool payloads and
prompt fields, NER catches what only shows up in prose. Any PERSON/GPE/
LOC-class entity found in agent-LLM-bound text is a failing test under
Q27's 'no PII, under any circumstances' rule, not a warning.").

Real, not mocked: every scenario is a genuine Decision Agent or Image
Parsing Agent LLM call (tests/eval/scenario_runners.py), and entity
extraction is a genuine Azure OpenAI GPT-4.1 mini call
(tests/eval/ner_utils.py) -- not a regex or a mocked NER library.

Scoped to each agent's final `reasoning` field specifically -- the one
piece of free text this project's structured-output schemas (Verdict,
ConsistencyAssessment) actually expose for prose to hide in. Real tool-
call arguments were considered too (the spec also lists them) but are
already covered by tests/test_security_and_pii.py's structured-payload
PII scan; re-scanning them here with a slower, real LLM call per argument
would duplicate that coverage rather than add new coverage, so it's left
out -- a real, deliberate scope narrowing, not silently dropped.

Image Parsing coverage re-added here (project-plan.md Q87) after being
pulled in Q86 -- a real, confirmed base64-truncation bug meant every real
Image Parsing scenario never produced a real reasoning text to scan in
the first place. The fix (Q86/Q87's analyze_claim_photo tool) landed, so
this is real coverage again, not a test that can only ever fail for a
reason unrelated to what it's meant to check.

Uses the same 5-scenario Decision Agent sample and 3-scenario Image
Parsing sample as tests/eval/similarity_entity_report.py /
reference_answers.py, so one real agent run per scenario id serves both
files' needs across a full pytest run, not two separate real-cost samples.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner_utils import extract_entities  # noqa: E402
from reference_answers import DECISION_REFERENCE_SCENARIOS, IMAGE_PARSING_REFERENCE_SCENARIOS  # noqa: E402
from scenario_runners import run_decision_reasoning, run_image_parsing_reasoning  # noqa: E402


@pytest.mark.parametrize(
    "scenario_id", [s.scenario_id for s in DECISION_REFERENCE_SCENARIOS], ids=lambda s: s,
)
def test_decision_agent_reasoning_has_no_pii_leak(scenario_id):
    reasoning = run_decision_reasoning(scenario_id)
    entities = extract_entities(reasoning)
    assert not entities.persons, f"scenario {scenario_id!r}: PERSON-class entity leaked into reasoning: {entities.persons}"
    assert not entities.locations, f"scenario {scenario_id!r}: GPE/LOC-class entity leaked into reasoning: {entities.locations}"


@pytest.mark.parametrize(
    "reference_scenario", IMAGE_PARSING_REFERENCE_SCENARIOS, ids=[s.reference_id for s in IMAGE_PARSING_REFERENCE_SCENARIOS],
)
def test_image_parsing_agent_reasoning_has_no_pii_leak(reference_scenario):
    reasoning = run_image_parsing_reasoning(reference_scenario)
    entities = extract_entities(reasoning)
    assert not entities.persons, (
        f"scenario {reference_scenario.reference_id!r}: PERSON-class entity leaked into reasoning: {entities.persons}"
    )
    assert not entities.locations, (
        f"scenario {reference_scenario.reference_id!r}: GPE/LOC-class entity leaked into reasoning: {entities.locations}"
    )
