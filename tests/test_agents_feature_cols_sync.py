"""agents/fraud_scoring_agent.py hardcodes a copy of ml/fraud_attribution.py's
FEATURE_COLS (see that file's comment for why -- importing fraud_attribution,
and therefore shap, into a process that already has crewai loaded crashes).
This test is what keeps that copy honest: it must import fraud_attribution
BEFORE crewai in this process, or it would hit the same collision it exists
to guard against.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from fraud_attribution import FEATURE_COLS  # noqa: E402 -- must precede crewai's import below

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.fraud_scoring_agent import _FRAUD_FEATURE_COLS  # noqa: E402


def test_fraud_scoring_agent_feature_cols_matches_real_model():
    assert tuple(FEATURE_COLS) == _FRAUD_FEATURE_COLS, (
        "agents/fraud_scoring_agent.py's hardcoded _FRAUD_FEATURE_COLS has drifted from "
        "ml/fraud_attribution.py's real FEATURE_COLS -- update the copy to match"
    )
