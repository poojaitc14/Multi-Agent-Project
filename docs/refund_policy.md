Version: v1.0

# Refund Policy

## Return window

Standard claims must be filed within **30 days** of delivery. Change of Mind claims have a shorter **14-day** sub-window and must be unused/unopened.

## Claim categories

### Damaged in Transit
Item arrived physically damaged. Photo evidence required. Eligible for full refund to the original payment method when within the return window and photo evidence is consistent with the claim.

### Wrong Item Received
Item doesn't match what was ordered. Photo evidence required. Eligible for full refund to the original payment method when within the return window and photo evidence is consistent with the claim.

### Not as Described
Item functions but materially differs from its listing. Photo evidence required. Customer's choice of full refund or store credit — the only category offering that choice.

### Defective/DOA
Item doesn't work out of the box. Photo evidence required. Eligible for full refund to the original payment method when within the return window and photo evidence is consistent with the claim.

### Change of Mind
No defect — the customer simply doesn't want the item. No photo required. Store credit only, not a refund to the original payment method. Item must be unused/unopened. Governed by the 14-day sub-window, not the standard 30-day window.

## Decision matrix

The Decision Agent combines the Image Parsing Agent's consistency verdict with the ML Fraud Scoring Agent's risk band to reach a verdict:

| Image verdict \ Fraud risk | Low | Medium | High |
|---|---|---|---|
| Consistent | Auto-approve | Auto-approve | Escalate |
| Partially consistent | Auto-approve | Escalate | Escalate |
| Inconsistent | Escalate | Auto-deny | Auto-deny |
| No photo provided (photo-required category) | Re-prompt customer for photo — no verdict yet | — | — |

## Guardrail

Any claim with a refund amount above **$200** always escalates to human review, regardless of the matrix outcome above. This guardrail overrides every other rule in this document — it is checked last and cannot be bypassed by any combination of consistency and risk.
