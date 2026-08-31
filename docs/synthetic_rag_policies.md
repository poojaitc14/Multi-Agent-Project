# Synthetic RAG Policy Corpus (15 documents)

Generated as content only — nothing here is wired into `search_refund_policy`, Chroma/OpenSearch, or any ingestion pipeline. Fifteen deliberately varied fictional retailer refund policies, meant as a richer synthetic corpus than the single canonical `docs/refund_policy.md` — useful for testing whether retrieval finds the *right* policy/clause among distractors, not just the only document that exists. Each carries its own semantic version per the project's versioning convention (Q35/Q46).

---

## 1. Northfield Apparel Co. — `v1.0`

**Return window:** 45 days from delivery.
**Categories:** Damaged in Transit, Wrong Item, Not as Described, Change of Mind (no defect).
**Refund form:** Damaged/Wrong Item → full refund, original payment. Not as Described → store credit only. Change of Mind → store credit, item must have tags attached, 30-day sub-window.
**Photo requirement:** Required for all defect-based claims; not required for Change of Mind.
**Guardrail:** Refunds over **$150** always escalate to a human reviewer.
**Notes:** Worn items with tags removed are ineligible regardless of claim category.

---

## 2. Circuit & Byte Electronics — `v2.3`

**Return window:** 30 days from delivery, 14 days for opened software/digital-license bundles.
**Categories:** Defective/DOA, Wrong Item, Damaged in Transit, Not as Described.
**Refund form:** All categories → full refund to original payment method only; no store credit offered.
**Photo requirement:** Required for all claims, plus a photo of the serial number.
**Guardrail:** Refunds over **$500** always escalate. Items over $1,000 additionally require a support ticket number referenced in the claim.
**Notes:** Restocking fee of 15% applies to Change of Mind returns on opened electronics (Change of Mind is otherwise not a covered category here).

---

## 3. Marketplace Direct (3rd-Party Seller Policy) — `v1.4`

**Return window:** Seller-defined, 7–60 days; defaults to 14 days if the seller hasn't specified one.
**Categories:** Item Not Received, Not as Described, Damaged in Transit, Counterfeit/Suspected Counterfeit.
**Refund form:** Platform mediates — refund issued from platform escrow, not directly from the seller, pending seller response window (5 business days).
**Photo requirement:** Required for Not as Described, Damaged, and Counterfeit claims.
**Guardrail:** Any Counterfeit claim escalates automatically regardless of amount. Refunds over **$100** also escalate.
**Notes:** Item Not Received claims are cross-checked against carrier tracking before any refund is considered.

---

## 4. FreshCart Grocery & Perishables — `v1.1`

**Return window:** 48 hours from delivery for perishables; 14 days for shelf-stable/non-food items.
**Categories:** Damaged in Transit, Wrong Item, Quality Issue (spoiled/expired on arrival).
**Refund form:** Perishables → refund only, no replacement or store credit (food-safety policy). Non-food → customer's choice of refund or store credit.
**Photo requirement:** Required for Quality Issue and Damaged claims; not required for Wrong Item.
**Guardrail:** Refunds over **$75** escalate — lower threshold than most categories, reflecting lower average order value.
**Notes:** No Change of Mind category exists for perishable goods.

---

## 5. Homestead Furniture & Living — `v1.0`

**Return window:** 60 days from delivery.
**Categories:** Damaged in Transit, Defective/DOA, Not as Described, Change of Mind.
**Refund form:** Damaged/Defective → full refund including original delivery fee. Not as Described → full refund, delivery fee not included. Change of Mind → refund minus a 20% restocking fee plus return shipping cost, item must be unassembled or reassembled to original condition.
**Photo requirement:** Required for all claims; large items require photos from at least 3 angles.
**Guardrail:** Refunds over **$400** escalate, given high average order values in this category.
**Notes:** Assembled furniture returned for Change of Mind is inspected on pickup; visible assembly damage voids the refund.

---

## 6. Lumière Beauty & Cosmetics — `v1.2`

**Return window:** 30 days, unopened only. Opened/used cosmetics are non-returnable for hygiene reasons except for Defective/DOA.
**Categories:** Defective/DOA (product spoiled, broken applicator, wrong formula sealed), Wrong Item, Damaged in Transit.
**Refund form:** Store credit by default; refund to original payment available on request within the first 14 days only.
**Photo requirement:** Required for Defective/DOA and Damaged in Transit.
**Guardrail:** Refunds over **$200** escalate.
**Notes:** No Change of Mind or Not as Described category — shade/scent mismatches are explicitly excluded from coverage.

---

## 7. StreamVault Digital Goods — `v1.0`

**Return window:** N/A for most digital purchases (all sales final once downloaded/streamed) — exception below.
**Categories:** Billing Error (duplicate/unauthorized charge), Technical Failure (content undeliverable due to platform fault).
**Refund form:** Full refund to original payment method only.
**Photo requirement:** Not applicable — claims require a transaction ID and, for Technical Failure, an error-log screenshot.
**Guardrail:** Refunds over **$50** escalate — deliberately low, since most digital-goods refund abuse involves small-dollar repeated claims.
**Notes:** This policy has no "escalate for ambiguous cases" middle tier — claims are either auto-approved against clear billing records or denied.

---

## 8. Petal & Paw Subscription Box — `v1.3`

**Return window:** Damage/defect claims within 10 days of delivery; subscription cancellation is separate from returns and not covered here.
**Categories:** Damaged in Transit, Wrong Item (incorrect box contents), Missing Item.
**Refund form:** Store credit toward next box by default; cash refund available for Damaged/Wrong Item within the first 3 subscription cycles only.
**Photo requirement:** Required for Damaged and Wrong Item; Missing Item requires a photo of the full unpacked box contents instead.
**Guardrail:** Refunds over **$60** escalate — reflects the low per-box price point of this category.
**Notes:** Repeated Missing Item claims (3+ in a rolling 90-day window) trigger manual review regardless of the guardrail amount.

---

## 9. Aurelia Fine Jewelry — `v2.0`

**Return window:** 30 days from delivery; custom/engraved pieces are final sale.
**Categories:** Damaged in Transit, Not as Described (materials/carat mismatch), Defective/DOA (clasp, setting failure).
**Refund form:** Full refund to original payment method; store credit not offered for this category.
**Photo requirement:** Required for all claims, plus the original certificate of authenticity if one was issued.
**Guardrail:** Refunds over **$1,000** escalate — a much higher threshold than most categories, appropriate to typical order values, but *any* claim on a certified/appraised piece escalates regardless of amount.
**Notes:** Items returned without original packaging and authenticity documentation are subject to a secondary authentication review before refund.

---

## 10. TrailWorks Auto Parts — `v1.1`

**Return window:** 90 days for unopened/uninstalled parts; 30 days for Defective/DOA on installed parts with proof of professional installation.
**Categories:** Wrong Item (incorrect fitment), Defective/DOA, Damaged in Transit.
**Refund form:** Full refund to original payment method; a core charge (for parts like alternators/starters) is refunded separately upon return of the old core.
**Photo requirement:** Required for all claims; Defective/DOA on installed parts additionally requires an installer's diagnostic note.
**Guardrail:** Refunds over **$300** escalate.
**Notes:** Parts returned showing signs of installation are only eligible under Defective/DOA, never under a general Change-of-Mind-style claim (which this policy doesn't offer at all).

---

## 11. Whisker & Wag Pet Supplies — `v1.0`

**Return window:** 30 days; opened pet food/treats are non-returnable unless Defective/DOA (contamination, spoilage).
**Categories:** Defective/DOA, Wrong Item, Damaged in Transit, Change of Mind (unopened only).
**Refund form:** Customer's choice of refund or store credit for all categories.
**Photo requirement:** Required for Defective/DOA and Damaged in Transit.
**Guardrail:** Refunds over **$150** escalate.
**Notes:** Recalled-product claims (matched against an internal recall list) bypass the standard matrix and auto-approve regardless of photo evidence.

---

## 12. Chapter & Verse Books and Media — `v1.5`

**Return window:** 21 days from delivery.
**Categories:** Damaged in Transit, Wrong Item, Not as Described (wrong edition/format).
**Refund form:** Full refund to original payment method for Damaged/Wrong Item; store credit for Not as Described.
**Photo requirement:** Required only for Damaged in Transit; Wrong Item and Not as Described can be resolved via ISBN/edition comparison against the order record without a photo.
**Guardrail:** Refunds over **$80** escalate — reflects low average order value.
**Notes:** This is the one policy in this corpus where a claim category (Wrong Item, Not as Described) is explicitly allowed to reach a verdict *without* photo evidence, since the mismatch is verifiable from order data alone.

---

## 13. Summit & Stride Sporting Goods — `v1.2`

**Return window:** 45 days; worn/used athletic footwear is non-returnable except Defective/DOA.
**Categories:** Defective/DOA, Damaged in Transit, Wrong Item, Change of Mind (unworn, original box).
**Refund form:** Damaged/Defective/Wrong Item → full refund. Change of Mind → store credit, 20-day sub-window, must include original shoebox in resellable condition.
**Photo requirement:** Required for all defect-based claims, including a close-up of any material failure point.
**Guardrail:** Refunds over **$250** escalate.
**Notes:** "Wardrobing" (returning visibly-used gear as unworn) is explicitly named as a fraud pattern this policy watches for via the packaging-condition check.

---

## 14. Ironclad Home Improvement & Tools — `v1.0`

**Return window:** 90 days for hand tools and hardware; 30 days for powered/electrical tools.
**Categories:** Defective/DOA, Damaged in Transit, Wrong Item.
**Refund form:** Full refund to original payment method; store credit available on request.
**Photo requirement:** Required for all claims; powered tools additionally require a photo of the safety label/model number.
**Guardrail:** Refunds over **$350** escalate. Any claim involving a tool under active safety recall auto-escalates regardless of amount.
**Notes:** No Change of Mind or Not as Described category — this retailer's policy is narrower in scope than most others in this corpus, covering only genuine defect/shipping-error claims.

---

## 15. Little Sprout Toys & Games — `v1.1`

**Return window:** 60 days from delivery, extended to 90 days from November 1 through December 31 for holiday purchases.
**Categories:** Defective/DOA, Damaged in Transit, Wrong Item, Change of Mind (unopened).
**Refund form:** Customer's choice of refund or store credit for all categories.
**Photo requirement:** Required for Defective/DOA and Damaged in Transit; not required for unopened Change of Mind or clear Wrong Item cases matched against the order record.
**Guardrail:** Refunds over **$120** escalate.
**Notes:** Choking-hazard/safety-recall claims bypass the standard matrix and auto-approve immediately, the same pattern as Whisker & Wag's recall handling above — the one behavior that repeats identically across two otherwise-unrelated policies in this corpus, useful as a retrieval test case (a query about recalls should surface *both* documents).
