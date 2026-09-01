"""Image Parsing Agent golden set (project-plan.md Q89/Q90) -- 25
hand-crafted scenarios built from real (not exclusively synthetic)
photos: the 6 Roboflow damaged/intact package photos and 3 real DummyJSON
product photos added in Q89, plus reuse of this project's existing 7
synthetic fixtures (`tests/fixtures/generate_test_images.py`), each
paired against a real DummyJSON order_ref.

Unlike golden_set.py's Decision Agent scenarios (checked against a
deterministic decision-matrix lookup), analyze_claim_photo's real verdict
is a genuine GPT-4.1 mini vision judgment -- there is no deterministic
"correct" verdict for every photo the way there is for the Decision
Agent's matrix. This file is honest about that distinction:

- **Hard-gating scenarios** (`expected_verdict`/`expected_product_match`
  set): only where the ground truth is genuinely, objectively knowable --
  no photo stored at all (deterministic `no_photo`, no LLM call even
  happens per analyze_claim_photo's own design), or a photo of a
  completely different real product than the one ordered (an objective
  mismatch, not a judgment call). A third case originally attempted here
  -- "the ordered product's own real photo" -- turned out NOT to be
  reliably knowable in advance either: a real run (project-plan.md Q90)
  found DummyJSON's own text description doesn't always accurately
  describe its own product photo, so even a genuinely correct photo can
  legitimately fail a text-description comparison. Demoted to
  informational rather than kept as a false hard guarantee.
- **Informational scenarios** (`expected_verdict`/`expected_product_match`
  left `None`): everything where the "correct" answer is itself a real
  judgment call this file has no honest way to know in advance -- does a
  generic cardboard box photo "match" a specific branded product, how
  damaged is "damaged enough" to read as `consistent` vs
  `partially_consistent`. Scored and reported for real by
  test_image_golden_set.py, never asserted against a fabricated ground
  truth.

Real DummyJSON order_ref -> product mappings used below (confirmed via
real `curl https://dummyjson.com/carts/{order_ref}` calls, not guessed):
1 -> "Blue Frock", 2 -> "Man Short Sleeve Shirt",
4 -> "Sports Sneakers Off White Red", 5 -> "Samsung Galaxy Tab White".
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_IMAGES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "images"


@dataclass
class ImageGoldenScenario:
    scenario_id: str
    photo_path: Optional[Path]  # None -> no photo stored for this claim at all
    order_ref: str
    claim_category: str
    claim_description: str
    expected_verdict: Optional[str] = None
    expected_product_match: Optional[bool] = None

    @property
    def claim_ref(self) -> str:
        return f"clm_imggolden_{self.scenario_id}"

    @property
    def is_hard_gated(self) -> bool:
        return self.expected_verdict is not None or self.expected_product_match is not None


IMAGE_GOLDEN_SET = [
    # --- Tier A: no photo stored -- deterministic, 3 scenarios ---
    ImageGoldenScenario(
        "no_photo_1", None, "1", "Damaged in Transit", "The box arrived crushed.",
        expected_verdict="no_photo", expected_product_match=False,
    ),
    ImageGoldenScenario(
        "no_photo_2", None, "2", "Wrong Item Received", "I got a completely different item.",
        expected_verdict="no_photo", expected_product_match=False,
    ),
    ImageGoldenScenario(
        "no_photo_3", None, "5", "Defective/DOA", "It doesn't turn on at all.",
        expected_verdict="no_photo", expected_product_match=False,
    ),
    # --- Tier B: photo of a genuinely different real product -- objective mismatch, 4 scenarios ---
    ImageGoldenScenario(
        "mismatch_homepod_vs_frock", _IMAGES_DIR / "real_wrong_item_homepod.jpg", "1",
        "Wrong Item Received", "I ordered a dress but got this instead.",
        expected_product_match=False,
    ),
    ImageGoldenScenario(
        "mismatch_homepod_vs_sneakers", _IMAGES_DIR / "real_wrong_item_homepod.jpg", "4",
        "Wrong Item Received", "I ordered sneakers but got this instead.",
        expected_product_match=False,
    ),
    ImageGoldenScenario(
        "mismatch_synthetic_vs_frock", _IMAGES_DIR / "wrong_item.jpg", "1",
        "Wrong Item Received", "This is not what I ordered.",
        expected_product_match=False,
    ),
    ImageGoldenScenario(
        "mismatch_synthetic_vs_sneakers", _IMAGES_DIR / "wrong_item.jpg", "4",
        "Wrong Item Received", "This is not what I ordered.",
        expected_product_match=False,
    ),
    # --- Tier C: photo of the genuine, correct real product -- informational, not hard-gated ---
    # A real run confirmed this tier's original premise was wrong: "the
    # real product's own real photo must read as product_match=True" isn't
    # actually a safe assumption. analyze_claim_photo compares the photo
    # against get_product_reference's TEXT description, not "is this
    # literally the stored photo" -- and DummyJSON's own description text
    # for "Blue Frock" ("vibrant blue color") doesn't accurately describe
    # its own real photo (a white dress with a blue polka-dot-style
    # pattern), a genuine third-party demo-data inconsistency this project
    # doesn't control. The real model's judgment (product_match=False,
    # citing the color/pattern mismatch against the text description) is
    # defensible given what it was actually asked to compare -- the flaw
    # was this file's assumption, not analyze_claim_photo. Kept
    # informational rather than removed, since the scenarios are still
    # real and worth observing.
    ImageGoldenScenario(
        "match_frock_defective", _IMAGES_DIR / "real_intact_blue_frock.jpg", "1",
        "Defective/DOA", "The zipper is broken and won't close.",
    ),
    ImageGoldenScenario(
        "match_sneakers_not_as_described", _IMAGES_DIR / "real_intact_sneakers.jpg", "4",
        "Not as Described", "The color looks different from the listing photos.",
    ),
    # --- Tier D: informational -- real photos, genuinely judgment-based ground truth, 16 scenarios ---
    ImageGoldenScenario(
        "damaged_item_o1", _IMAGES_DIR / "damaged_item.jpg", "1",
        "Damaged in Transit", "The box arrived crushed and the item inside is visibly cracked.",
    ),
    ImageGoldenScenario(
        "damaged_item_o5", _IMAGES_DIR / "damaged_item.jpg", "5",
        "Damaged in Transit", "The box arrived crushed and the item inside is visibly cracked.",
    ),
    ImageGoldenScenario(
        "damaged_shoes_o4", _IMAGES_DIR / "damaged_shoes.jpg", "4",
        "Damaged in Transit", "The shoes arrived damaged, with the sole visibly separated from the upper.",
    ),
    ImageGoldenScenario(
        "damaged_shoes_o1", _IMAGES_DIR / "damaged_shoes.jpg", "1",
        "Damaged in Transit", "The shoes arrived damaged, with the sole visibly separated from the upper.",
    ),
    ImageGoldenScenario(
        "damaged_clothes_o1", _IMAGES_DIR / "damaged_clothes.jpg", "1",
        "Damaged in Transit", "The hoodie arrived with a torn seam and a stain.",
    ),
    ImageGoldenScenario(
        "damaged_headphones_o5", _IMAGES_DIR / "damaged_headphones.jpg", "5",
        "Defective/DOA", "The headphones arrived with a cracked ear cup.",
    ),
    ImageGoldenScenario(
        "intact_item_o2", _IMAGES_DIR / "intact_item.jpg", "2",
        "Not as Described", "The material feels different than what was described.",
    ),
    ImageGoldenScenario(
        "shipping_label_o1", _IMAGES_DIR / "shipping_label_photo.jpg", "1",
        "Damaged in Transit", "The package arrived damaged.",
    ),
    ImageGoldenScenario(
        "roboflow_damaged_box1_o1", _IMAGES_DIR / "roboflow_damaged_box_1.jpg", "1",
        "Damaged in Transit", "The box was crushed in shipping.",
    ),
    ImageGoldenScenario(
        "roboflow_damaged_box1_o4", _IMAGES_DIR / "roboflow_damaged_box_1.jpg", "4",
        "Damaged in Transit", "The box was crushed in shipping.",
    ),
    ImageGoldenScenario(
        "roboflow_damaged_box2_o2", _IMAGES_DIR / "roboflow_damaged_box_2.jpg", "2",
        "Damaged in Transit", "The package box arrived visibly damaged.",
    ),
    ImageGoldenScenario(
        "roboflow_damaged_foodbox1_o4", _IMAGES_DIR / "roboflow_damaged_food_box_1.jpg", "4",
        "Damaged in Transit", "The package box was damaged on arrival.",
    ),
    ImageGoldenScenario(
        "roboflow_damaged_foodbox2_o5", _IMAGES_DIR / "roboflow_damaged_food_box_2.jpg", "5",
        "Damaged in Transit", "The package box was damaged on arrival.",
    ),
    ImageGoldenScenario(
        "roboflow_intact_foodbox1_o1", _IMAGES_DIR / "roboflow_intact_food_box_1.jpg", "1",
        "Not as Described", "The box looks fine but the contents aren't what I expected.",
    ),
    ImageGoldenScenario(
        "roboflow_intact_foodbox2_o2", _IMAGES_DIR / "roboflow_intact_food_box_2.jpg", "2",
        "Defective/DOA", "The item inside doesn't work.",
    ),
    ImageGoldenScenario(
        "roboflow_intact_foodbox2_o5", _IMAGES_DIR / "roboflow_intact_food_box_2.jpg", "5",
        "Defective/DOA", "The item inside doesn't work.",
    ),
]

assert len(IMAGE_GOLDEN_SET) == 25, f"expected 25 scenarios, got {len(IMAGE_GOLDEN_SET)}"
