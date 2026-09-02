"""RAGAS retrieval-quality evaluation (testing-and-evaluation-plan.md's
"RAG evaluation" section, Q22) -- scores search_refund_policy's real
retrieval and the Decision Agent's real verdict rationale on 4 metrics:
context precision, context recall, faithfulness (a.k.a. groundedness --
does the rationale stick to what was actually retrieved), and answer
relevancy. Also reports tool routing accuracy and tool call sequence
(testing-and-evaluation-plan.md Q9) from the same real runs, since this
file already produces exactly the TaskOutput.messages trace those need --
no reason to re-run the same 5 real scenarios a second time in a separate
script just to measure them.

Context recall (previously deferred -- flagged in project-plan.md Q83 as
needing a hand-written reference answer per scenario) reuses
reference_answers.py's DECISION_REFERENCE_SCENARIOS: RAGAS's ContextRecall
metric (confirmed via direct introspection of this project's installed
ragas==0.4.3, not assumed from docs: `ascore(user_input, retrieved_contexts,
reference)`) decomposes a REFERENCE ANSWER into claims and checks how many
are supported by retrieved_contexts.

Q102 revision: context_recall now scores against `policy_only_reference`
(a second, dedicated field on DecisionReferenceScenario), not the general-
purpose `reference_rationale` shared with the semantic-similarity metric.
The first real run found context_recall averaging 10% even though
context_precision was 96.7% -- retrieval was genuinely fine; the reference
just wasn't policy-only, so its true-but-non-policy claims ("fraud risk is
low") were correctly marked unsupported by a metric that only ever checks
against retrieved policy text. Faithfulness had the same root cause from
the response side, fixed by scoring it against `widened_contexts` (the
real policy chunks plus a synthetic summary of the claim's own given
facts) instead of `retrieved_contexts` alone -- context_precision still
scores against the real, unmodified `retrieved_contexts`, since widening
that one would stop it from measuring actual retrieval quality.

Real, not mocked: every scenario makes a genuine search_refund_policy call
against the real OpenSearch Serverless collection (project-plan.md Q81),
a genuine Decision Agent LLM call for the rationale being scored, and
genuine ragas LLM-judge + embedding calls to produce each metric.

Reuses this project's already-approved Azure OpenAI deployments -- GPT-4.1
mini as the RAGAS judge LLM, text-embedding-3-small as the embedder (the
same deployment ml/rag/embed_and_index.py already uses to index the policy
chunks) -- rather than adding a new provider. This is a different call
than testing-and-evaluation-plan.md's Q17 (semantic-similarity/NER on
customer-claim free text, which needs its own PII review before a
provider is picked): docs/refund_policy.md's chunks are static company
policy text, not customer data, so reusing GPT-4.1 mini/text-embedding-
3-small here doesn't touch that open question.

Purely informational output, like cost_report.py -- not a pass/fail gate;
testing-and-evaluation-plan.md's Q18/Q22 explicitly defer setting
thresholds until after a real score distribution exists to calibrate
against, which is exactly what running this produces.

Scored against a sample of golden_set.py's scenarios (not all 25) --
each scenario costs a real Decision Agent call plus several real RAGAS
judge/embedding calls, and this file's purpose is producing real,
trustworthy numbers to calibrate thresholds from, not maximizing scenario
count in one pass.

Run with: uv run python tests/eval/ragas_report.py
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agents.decision_agent import build_decision_agent, build_verdict_task  # noqa: E402
from agents.mcp_tools import decision_mcp_adapter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_set import EXPECTED_DECISION_AGENT_TOOLS, GOLDEN_SET, tool_call_sequence  # noqa: E402
from reference_answers import DECISION_REFERENCE_SCENARIOS  # noqa: E402

from crewai import Crew  # noqa: E402
from ragas.embeddings import embedding_factory  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithoutReference, ContextRecall, Faithfulness  # noqa: E402

_REFERENCE_BY_SCENARIO_ID = {ref.scenario_id: ref for ref in DECISION_REFERENCE_SCENARIOS}

# A representative sample, not the full golden set -- see module docstring.
SAMPLE_SCENARIO_IDS = {
    "consistent_low_00", "partially_consistent_high_10", "inconsistent_low_12",
    "change_of_mind_medium", "guardrail_override_consistent_low",
}


def _async_azure_client() -> AsyncAzureOpenAI:
    # RAGAS's .ascore() methods require an async-capable client -- confirmed
    # for real: passing the sync AzureOpenAI client here raised
    # "Cannot use agenerate() with a synchronous client" on the first
    # real run.
    return AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def run_real_scenario(scenario) -> dict:
    """Real search_refund_policy call (via a real Decision Agent tool
    call, not called directly, so retrieved_contexts reflects exactly
    what the agent actually saw) + real Decision Agent verdict."""
    with decision_mcp_adapter() as tools:
        search_tool = next(t for t in tools if t.name == "search_refund_policy")
        query = f"{scenario.claim_category}: {scenario.claim_description}"
        retrieval = search_tool.run(query=query)
        import json

        retrieval_data = json.loads(retrieval) if isinstance(retrieval, str) else retrieval
        retrieved_contexts = [c["text"] for c in retrieval_data.get("chunks", [])]

        agent = build_decision_agent(tools, customer_ref="cst_ragas_eval", claim_ref=scenario.claim_ref)
        task = build_verdict_task(
            agent, claim_ref=scenario.claim_ref, order_ref="1", claim_category=scenario.claim_category,
            claim_description=scenario.claim_description, refund_amount_usd=scenario.refund_amount_usd,
            image_verdict=scenario.image_verdict, fraud_risk_band=scenario.fraud_risk_band,
            fraud_key_signals=scenario.fraud_key_signals,
        )
        result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()

    messages = result.tasks_output[0].messages
    sequence = tool_call_sequence(messages)
    called_tools = set(sequence)
    routing_ok = called_tools <= EXPECTED_DECISION_AGENT_TOOLS and "apply_decision_matrix" in called_tools
    sequence_ok = (
        "search_refund_policy" not in sequence
        or "apply_decision_matrix" not in sequence
        or sequence.index("search_refund_policy") < sequence.index("apply_decision_matrix")
    )

    # project-plan.md Q102: the Decision Agent's real reasoning legitimately
    # cites real facts (image verdict, fraud risk band, refund amount) that
    # were genuinely given to it as task inputs, not retrieved via
    # search_refund_policy -- Faithfulness previously only ever saw the
    # policy chunks as "context", so every one of those true, given-fact
    # statements was correctly-by-RAGAS's-own-definition marked unsupported.
    # This widened context represents everything the agent legitimately had
    # access to, not just the retrieval subset -- used for faithfulness
    # only, never context_precision (which must stay scoped to what
    # search_refund_policy actually retrieved, or it stops measuring
    # retrieval quality at all).
    claim_facts_context = (
        f"Claim context (given, not retrieved): image consistency verdict is "
        f"'{scenario.image_verdict}', fraud risk band is '{scenario.fraud_risk_band}', "
        f"refund amount is ${scenario.refund_amount_usd:.2f}."
    )

    return {
        "user_input": query,
        "response": result.pydantic.reasoning,
        "retrieved_contexts": retrieved_contexts,
        "widened_contexts": retrieved_contexts + [claim_facts_context],
        "reference": _REFERENCE_BY_SCENARIO_ID[scenario.scenario_id].reference_rationale,
        "policy_only_reference": _REFERENCE_BY_SCENARIO_ID[scenario.scenario_id].policy_only_reference,
        "routing_ok": routing_ok,
        "sequence_ok": sequence_ok,
        "called_tools": sorted(called_tools),
    }


async def score_scenario(scenario_id, real, context_precision, context_recall, faithfulness, answer_relevancy) -> dict:
    # context_precision: real retrieved_contexts only -- must stay scoped to
    # what search_refund_policy actually returned, or it stops measuring
    # retrieval quality.
    precision_result = await context_precision.ascore(
        user_input=real["user_input"], response=real["response"], retrieved_contexts=real["retrieved_contexts"],
    )
    # context_recall: policy_only_reference (Q102), not the general-purpose
    # reference_rationale shared with the similarity metric -- so recall
    # measures "did retrieval get the right policy content", not "does the
    # reference's non-policy facts happen to appear in policy chunks".
    recall_result = await context_recall.ascore(
        user_input=real["user_input"], reference=real["policy_only_reference"],
        retrieved_contexts=real["retrieved_contexts"],
    )
    # faithfulness: widened_contexts (Q102) -- includes the real, given
    # claim-context facts the response legitimately draws from, not just
    # the policy subset.
    faithfulness_result = await faithfulness.ascore(
        user_input=real["user_input"], response=real["response"], retrieved_contexts=real["widened_contexts"],
    )
    relevancy_result = await answer_relevancy.ascore(user_input=real["user_input"], response=real["response"])
    return {
        "scenario_id": scenario_id,
        "context_precision": precision_result.value,
        "context_recall": recall_result.value,
        "groundedness": faithfulness_result.value,  # faithfulness, in RAGAS's own naming
        "answer_relevancy": relevancy_result.value,
        "num_retrieved_chunks": len(real["retrieved_contexts"]),
        "routing_ok": real["routing_ok"],
        "sequence_ok": real["sequence_ok"],
        "called_tools": real["called_tools"],
    }


async def score_all(real_by_scenario: dict, context_precision, context_recall, faithfulness, answer_relevancy) -> list[dict]:
    rows = []
    for scenario_id, real in real_by_scenario.items():
        row = await score_scenario(scenario_id, real, context_precision, context_recall, faithfulness, answer_relevancy)
        rows.append(row)
        print(
            f"{row['scenario_id']:<45} precision={row['context_precision']:.3f} "
            f"recall={row['context_recall']:.3f} groundedness={row['groundedness']:.3f} "
            f"relevancy={row['answer_relevancy']:.3f} chunks={row['num_retrieved_chunks']} "
            f"routing={'OK' if row['routing_ok'] else 'FAIL'} sequence={'OK' if row['sequence_ok'] else 'FAIL'}"
        )
    return rows


def main() -> None:
    sample = [s for s in GOLDEN_SET if s.scenario_id in SAMPLE_SCENARIO_IDS]
    print(f"Running {len(sample)} real scenarios (search_refund_policy + Decision Agent) before RAGAS scoring...\n")

    # The real CrewAI calls run synchronously, outside any event loop --
    # crewai's own agent.execute_task() refuses to run sync from inside a
    # running event loop (confirmed for real: this is exactly the error
    # hit when this used to be structured as one big async function),
    # correctly, since it would otherwise silently reenter/block the loop.
    real_by_scenario = {scenario.scenario_id: run_real_scenario(scenario) for scenario in sample}

    async_client = _async_azure_client()
    judge_llm = llm_factory(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], provider="openai", client=async_client,
        temperature=0, max_tokens=2000,  # the default was confirmed too low: a real faithfulness call
        # hit IncompleteOutputException (max_tokens length limit) mid-run before this was added.
    )
    embedder = embedding_factory(
        provider="openai", model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"], client=async_client,
    )
    context_precision = ContextPrecisionWithoutReference(llm=judge_llm)
    context_recall = ContextRecall(llm=judge_llm)
    faithfulness = Faithfulness(llm=judge_llm)
    answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=embedder)

    print("\nScoring with RAGAS (context precision, context recall, groundedness/faithfulness, answer relevancy)...\n")
    rows = asyncio.run(score_all(real_by_scenario, context_precision, context_recall, faithfulness, answer_relevancy))

    print("\nReal score distribution (for threshold calibration, testing-and-evaluation-plan.md Q18/Q22):")
    for metric in ("context_precision", "context_recall", "groundedness", "answer_relevancy"):
        values = [r[metric] for r in rows]
        print(f"  {metric}: min={min(values):.3f} max={max(values):.3f} avg={sum(values) / len(values):.3f}")

    routing_pass = sum(1 for r in rows if r["routing_ok"])
    sequence_pass = sum(1 for r in rows if r["sequence_ok"])
    print(f"\nTool routing accuracy: {routing_pass}/{len(rows)} scenarios called only the expected tool set")
    print(f"Tool call sequence accuracy: {sequence_pass}/{len(rows)} scenarios called search_refund_policy before apply_decision_matrix")
    for r in rows:
        if not r["routing_ok"] or not r["sequence_ok"]:
            print(f"  {r['scenario_id']}: called_tools={r['called_tools']} routing_ok={r['routing_ok']} sequence_ok={r['sequence_ok']}")


if __name__ == "__main__":
    main()
