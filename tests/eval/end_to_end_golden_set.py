"""End-to-end golden dataset (project-plan.md Q90 -- user-added, distinct
from golden_set.py and image_golden_set.py): 20 hand-crafted customer
scenarios run for REAL through the actual customer-facing backend
endpoints (`POST /messages`, `POST /messages/photo`) -- the same real
pipeline `chat/app.py` drives -- capturing the genuine outputs into a JSON
file.

Not a pass/fail gate, and deliberately not asserting expected outcomes:
the full pipeline has genuine, already-documented non-determinism this
project doesn't paper over -- Fraud Scoring's risk profile is bootstrap-
sampled per new customer_ref (mcp-servers/orchestrator_server.py's
get_account_info docstring), and Image Parsing's vision judgment is a
real GPT-4.1 mini call, not a deterministic lookup (see
image_golden_set.py's own module docstring for the same reasoning). This
file's job is producing a real, honest snapshot of what the whole project
actually does end to end for 20 realistic claims -- a reference dataset,
not a re-runnable exact-match test suite (running it again would likely
produce genuinely different fraud_risk_band/image_verdict values for the
same inputs, which is expected, not a bug).

Real, not mocked: every scenario is a genuine HTTP round trip against a
running `backend` (requires `docker compose up -d` or an equivalent local
run first) -- real Orchestrator intake, real Image Parsing vision calls
where a photo is attached, real Fraud Scoring, real Decision Agent, real
Postgres/DynamoDB/S3 underneath.

Run with: uv run python tests/eval/end_to_end_golden_set.py
Writes: tests/eval/end_to_end_golden_dataset.json
"""

import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

BACKEND_URL = "http://127.0.0.1:8002"
_IMAGES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "images"


@dataclass
class EndToEndScenario:
    scenario_id: str
    order_ref: str
    claim_category: str
    claim_description: str
    days_to_return: int
    photo_path: Optional[Path] = None
    send_follow_up_after_photo: bool = True  # False -> deliberately stop after re_prompt_for_photo


SCENARIOS = [
    EndToEndScenario("change_of_mind_frock", "1", "Change of Mind", "I don't want this Blue Frock anymore, it's still unopened.", 2),
    EndToEndScenario("change_of_mind_shirt", "2", "Change of Mind", "Changed my mind about this shirt, unworn and still tagged.", 5),
    EndToEndScenario("damaged_sneakers", "4", "Damaged in Transit", "The sneakers arrived damaged, the sole is coming apart.", 3, _IMAGES_DIR / "damaged_shoes.jpg"),
    EndToEndScenario("damaged_frock", "1", "Damaged in Transit", "The dress arrived with a visible tear.", 1, _IMAGES_DIR / "damaged_item.jpg"),
    EndToEndScenario("damaged_tablet", "5", "Damaged in Transit", "The tablet box arrived crushed and damaged.", 4, _IMAGES_DIR / "roboflow_damaged_food_box_2.jpg"),
    EndToEndScenario("wrong_item_frock", "1", "Wrong Item Received", "I ordered a dress but received a completely different item.", 2, _IMAGES_DIR / "real_wrong_item_homepod.jpg"),
    EndToEndScenario("wrong_item_sneakers", "4", "Wrong Item Received", "This is not the sneakers I ordered.", 3, _IMAGES_DIR / "wrong_item.jpg"),
    EndToEndScenario("not_as_described_shirt", "2", "Not as Described", "The material feels completely different from the listing.", 6, _IMAGES_DIR / "intact_item.jpg"),
    EndToEndScenario("not_as_described_sneakers", "4", "Not as Described", "The color looks different from the listing photos.", 4, _IMAGES_DIR / "real_intact_sneakers.jpg"),
    EndToEndScenario("defective_tablet", "5", "Defective/DOA", "The tablet doesn't power on at all.", 1, _IMAGES_DIR / "damaged_headphones.jpg"),
    EndToEndScenario("defective_frock", "1", "Defective/DOA", "The zipper is broken and won't close.", 2, _IMAGES_DIR / "real_intact_blue_frock.jpg"),
    EndToEndScenario("damaged_shirt", "2", "Damaged in Transit", "The shirt box arrived crushed in shipping.", 3, _IMAGES_DIR / "roboflow_damaged_box_1.jpg"),
    EndToEndScenario("damaged_watch", "6", "Damaged in Transit", "The watch box arrived visibly damaged.", 2, _IMAGES_DIR / "roboflow_damaged_box_2.jpg"),
    EndToEndScenario("wrong_item_tablet", "5", "Wrong Item Received", "I received a completely different product.", 5, _IMAGES_DIR / "real_wrong_item_homepod.jpg"),
    EndToEndScenario("not_as_described_frock", "1", "Not as Described", "The box looks fine but the contents aren't what I expected.", 4, _IMAGES_DIR / "roboflow_intact_food_box_1.jpg"),
    EndToEndScenario("defective_sneakers", "4", "Defective/DOA", "The sneakers arrived damaged, with the sole visibly separated.", 3, _IMAGES_DIR / "damaged_shoes.jpg"),
    EndToEndScenario("change_of_mind_sneakers", "4", "Change of Mind", "I just don't want these sneakers anymore, unused and unopened.", 1),
    EndToEndScenario(
        "damaged_frock_never_sends_photo", "1", "Damaged in Transit", "The box arrived crushed.", 2,
        photo_path=None, send_follow_up_after_photo=False,
    ),
    EndToEndScenario("defective_shirt", "2", "Defective/DOA", "The shirt arrived with a torn seam and a stain.", 4, _IMAGES_DIR / "damaged_clothes.jpg"),
    EndToEndScenario("wrong_item_watch", "6", "Wrong Item Received", "I received a completely different product.", 6, _IMAGES_DIR / "roboflow_intact_food_box_2.jpg"),
]

assert len(SCENARIOS) == 20, f"expected 20 scenarios, got {len(SCENARIOS)}"


def run_scenario(scenario: EndToEndScenario) -> dict:
    customer_identifier = f"e2e-golden-{scenario.scenario_id}-{uuid.uuid4().hex[:6]}@example.com"
    result: dict = {
        "scenario_id": scenario.scenario_id,
        "customer_identifier": customer_identifier,
        "order_ref": scenario.order_ref,
        "claim_category": scenario.claim_category,
        "claim_description": scenario.claim_description,
        "days_to_return": scenario.days_to_return,
        "photo_attached": scenario.photo_path is not None,
        "turns": [],
    }

    first_message = (
        f"Order {scenario.order_ref}, {scenario.claim_category}: {scenario.claim_description} "
        f"It was {scenario.days_to_return} day(s) ago."
    )
    resp = requests.post(
        f"{BACKEND_URL}/messages", json={"customer_identifier": customer_identifier, "message": first_message}, timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    result["turns"].append({"sent": first_message, "response": data})

    claim_result = data.get("claim_result")
    if claim_result and claim_result.get("outcome") == "re_prompt_for_photo" and scenario.photo_path is not None:
        # /messages/photo takes real multipart form data, not base64 JSON -- matches chat/app.py's real call shape.
        photo_resp = requests.post(
            f"{BACKEND_URL}/messages/photo",
            data={"customer_identifier": customer_identifier},
            files={"photo": (scenario.photo_path.name, scenario.photo_path.read_bytes(), "image/jpeg")},
            timeout=30,
        )
        photo_resp.raise_for_status()
        result["turns"].append({"sent": f"[attached photo: {scenario.photo_path.name}]", "response": photo_resp.json()})

        if scenario.send_follow_up_after_photo:
            follow_up = "Here is the photo you asked for."
            resp2 = requests.post(
                f"{BACKEND_URL}/messages", json={"customer_identifier": customer_identifier, "message": follow_up}, timeout=180,
            )
            resp2.raise_for_status()
            data2 = resp2.json()
            result["turns"].append({"sent": follow_up, "response": data2})
            claim_result = data2.get("claim_result")

    result["final_claim_result"] = claim_result
    return result


def main() -> None:
    rows = []
    for i, scenario in enumerate(SCENARIOS):
        print(f"[{i + 1}/{len(SCENARIOS)}] running {scenario.scenario_id}...")
        try:
            row = run_scenario(scenario)
        except Exception as e:  # noqa: BLE001 -- a real failure here is data worth capturing, not hiding
            row = {"scenario_id": scenario.scenario_id, "error": f"{type(e).__name__}: {e}"}
            print(f"  ERROR: {row['error']}")
        else:
            outcome = (row.get("final_claim_result") or {}).get("outcome")
            print(f"  outcome={outcome!r}")
        rows.append(row)
        time.sleep(1)

    out_path = Path(__file__).resolve().parent / "end_to_end_golden_dataset.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {len(rows)} real scenario results to {out_path}")

    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"\n{len(errors)} scenario(s) hit a real error (see JSON for details): {[r['scenario_id'] for r in errors]}")


if __name__ == "__main__":
    main()
