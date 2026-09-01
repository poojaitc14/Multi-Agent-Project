"""Cost tracking (testing-and-evaluation-plan.md's "Agent evaluation
metrics" -- "Cost per claim" row, gate policy resolved by that document's
Q19: purely informational, no automated gate/ceiling, consistent with
project-plan.md Q23's "don't optimize for cost" stance).

Not a pytest test -- there's nothing to assert against, by design (Q19).
A real, standalone report: queries Langfuse's real REST API for every
golden-set scenario's real trace (test_golden_set.py's real Decision Agent
runs are already tagged with sessionId=claim_ref via agents/observability.py,
so every golden-set run this project does from now on is automatically
cost-tracked here with no extra instrumentation) and prints real
per-scenario and aggregate $ cost. `totalCost` comes straight from
Langfuse's own trace object -- not recomputed from raw token counts, since
Langfuse already does that computation against real, current model pricing.

Run with: uv run python tests/eval/cost_report.py
(after a real tests/eval/test_golden_set.py run has produced traces --
this only reports on runs that already happened, it doesn't trigger one)
"""

import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_set import GOLDEN_SET  # noqa: E402

LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")


def fetch_trace_cost(claim_ref: str, max_retries: int = 5) -> dict | None:
    """One real Langfuse API call per scenario -- claim_ref is the real
    sessionId every golden-set Decision Agent call is tagged with
    (build_decision_agent(tools, customer_ref, claim_ref) ->
    trace_metadata() -> additional_params, agents/observability.py).

    Retries on a real 429 -- confirmed hit empirically querying 25
    scenarios back-to-back against Langfuse Cloud's free tier -- honoring
    Retry-After when the response provides one rather than guessing a
    backoff, per Q30/Q73's established "this is a real, flaky external
    dependency, handle it, don't fabricate past it" pattern."""
    for attempt in range(max_retries):
        response = requests.get(
            f"{LANGFUSE_HOST}/api/public/traces",
            auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
            params={"sessionId": claim_ref, "limit": 1},
            timeout=30,
        )
        if response.status_code == 429:
            wait_s = float(response.headers.get("Retry-After", 5))
            time.sleep(wait_s)
            continue
        response.raise_for_status()
        data = response.json().get("data", [])
        return data[0] if data else None
    raise RuntimeError(f"Langfuse kept rate-limiting after {max_retries} retries for sessionId={claim_ref!r}")


def main() -> None:
    if not (LANGFUSE_HOST and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        print("Langfuse not configured (LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY) -- nothing to report.")
        return

    rows = []
    missing = []
    for scenario in GOLDEN_SET:
        trace = fetch_trace_cost(scenario.claim_ref)
        time.sleep(0.5)  # proactive spacing -- the 429 above is real, not hypothetical
        if trace is None:
            missing.append(scenario.scenario_id)
            continue
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "claim_category": scenario.claim_category,
                "cost_usd": trace.get("totalCost") or 0.0,
                "latency_ms": trace.get("latency") or 0.0,
                "num_observations": len(trace.get("observations") or []),
            }
        )

    if not rows:
        print("No real traces found for any golden-set scenario -- run tests/eval/test_golden_set.py first.")
        return

    print(f"{'scenario_id':<45} {'category':<22} {'$ cost':>10} {'latency(s)':>12} {'# LLM calls':>12}")
    total_cost = 0.0
    for row in sorted(rows, key=lambda r: -r["cost_usd"]):
        total_cost += row["cost_usd"]
        print(
            f"{row['scenario_id']:<45} {row['claim_category']:<22} "
            f"${row['cost_usd']:>9.5f} {row['latency_ms'] / 1000:>11.2f}s {row['num_observations']:>12}"
        )

    print(f"\n{len(rows)}/{len(GOLDEN_SET)} scenarios found real traces for.")
    print(f"Total real cost across these {len(rows)} traced scenarios: ${total_cost:.4f}")
    print(f"Average real cost per claim: ${total_cost / len(rows):.5f}")
    if missing:
        print(f"\nNo trace found (yet, or Langfuse's async ingestion hasn't caught up) for: {missing}")


if __name__ == "__main__":
    main()
