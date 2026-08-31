"""Feature attribution for score_fraud_risk (project-plan.md Q34) --
grounds FraudAssessment.key_signals in the real registered model instead
of letting the agent invent a plausible-sounding reason.

SHAP against the real MLflow-registered "fraud-risk-scoring" model
(v13, sigmoid-calibrated Logistic Regression, Slice 2), with a timing
check: Q34 explicitly allows falling back to the model's own
coefficients if SHAP proves too slow for a CPU-only, per-claim tool call.

Explains in the TRANSFORMED feature space (post-ColumnTransformer: scaled
numerics, one-hot categoricals), not raw columns -- SHAP's default masker
can't compare raw mixed string/numeric/bool data (confirmed empirically:
it does np.isclose() on everything, which breaks on strings). Transformed
names like "claim_category_Change of Mind" are still real, traceable
feature names, just more granular than the raw column name.
"""

import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Absolute, not relative -- relative paths broke the moment this got
# imported from mcp-servers/ instead of run directly from the repo root.
_ML_DIR = Path(__file__).resolve().parent
MLFLOW_TRACKING_URI = f"sqlite:///{(_ML_DIR / 'mlruns' / 'mlflow.db').as_posix()}"
REGISTRY_NAME = "fraud-risk-scoring"

# A small (82KB), self-contained, committed export of the registered
# production model (project-plan.md's Dockerfiles slice) -- ml/mlruns
# itself is 434MB, gitignored, and machine-local (the full experiment
# tracking store, not just the one model actually served), so it can't be
# baked into a reproducible Docker image. Re-export after promoting a new
# production version:
#   mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#   model = mlflow.sklearn.load_model(f"models:/{REGISTRY_NAME}@production")
#   mlflow.sklearn.save_model(model, str(_ML_DIR / "production_model"),
#       skops_trusted_types=["sklearn.calibration._CalibratedClassifier",
#                             "sklearn.calibration._SigmoidCalibration"])
# (the skops_trusted_types are needed because the registered model is a
# CalibratedClassifierCV-wrapped Logistic Regression -- confirmed for real,
# not guessed, by the export failing without them and naming exactly these
# two types.) Verified byte-identical predict_proba output against the
# registry-loaded model on 10 real sample rows before switching load_model()
# over to this path.
PRODUCTION_MODEL_PATH = str(_ML_DIR / "production_model")
DATA_PATH = str(_ML_DIR / "data" / "synthetic_fraud_risk_dataset.csv")
RANDOM_STATE = 42  # matches ml/model_training.ipynb exactly

NUMERIC_COLS = [
    "account_age_days", "total_orders_lifetime", "total_returns_lifetime", "claim_frequency_90d",
    "refund_amount_usd", "days_to_return", "customer_support_contacts_90d", "previous_dispute_count",
]
BOOL_COLS = ["address_match", "is_high_value_item", "photo_evidence_provided"]
CATEGORICAL_COLS = ["claim_category", "image_consistency"]
FEATURE_COLS = NUMERIC_COLS + BOOL_COLS + CATEGORICAL_COLS
CLASS_NAMES = ["high", "low", "medium"]  # LabelEncoder's sorted-alphabetical order, confirmed in Slice 2


def load_model():
    return mlflow.sklearn.load_model(PRODUCTION_MODEL_PATH)


def build_and_fit_transformer() -> ColumnTransformer:
    """The registered model was trained on ml/model_training.ipynb's
    X_train_transformed, not raw columns -- the ColumnTransformer itself
    was never saved as an MLflow artifact. Rebuilding + refitting it here
    on the exact same split (same file, same random_state, same
    train_test_split calls) reproduces an equivalent fitted transformer,
    since none of that is random beyond the seeded split."""
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df["fraud_risk_label"]
    X_trainval, _, y_trainval, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_train, _, _, _ = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=y_trainval)

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLS),
            ("boolean", "passthrough", BOOL_COLS),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    transformer.fit(X_train)
    return transformer


def load_background(transformer: ColumnTransformer, n: int = 50) -> np.ndarray:
    """A small real sample, transformed, to serve as SHAP's reference
    distribution -- not the claim being explained, just typical feature
    values to compare against."""
    df = pd.read_csv(DATA_PATH)
    sample = df[FEATURE_COLS].sample(n=n, random_state=42)
    return transformer.transform(sample)


def build_explainer(model, background: np.ndarray) -> shap.Explainer:
    return shap.Explainer(model.predict_proba, background)


def top_contributing_features(
    explainer: shap.Explainer, transformed_row: np.ndarray, feature_names: list[str], predicted_class_idx: int, top_n: int = 3
) -> list[dict]:
    shap_values = explainer(transformed_row)
    values_for_class = shap_values.values[0, :, predicted_class_idx]
    ranked = sorted(zip(feature_names, values_for_class, transformed_row[0]), key=lambda kv: abs(kv[1]), reverse=True)
    return [
        {"feature": name, "shap_value": float(val), "feature_value": float(fv)} for name, val, fv in ranked[:top_n]
    ]


class FraudAttributor:
    """What score_fraud_risk will eventually hold onto for the lifetime of
    the MCP server process -- model/transformer/explainer built once
    (that's the ~5.8s cost), reused per call (that's the ~0.01-0.02s cost).
    Building one of these per request would pay the warmup cost every time;
    this class exists specifically so that doesn't happen."""

    def __init__(self):
        self.model = load_model()
        self.transformer = build_and_fit_transformer()
        self.feature_names = list(self.transformer.get_feature_names_out())
        self.explainer = build_explainer(self.model, load_background(self.transformer))

    def score(self, raw_features: dict, top_n: int = 3) -> dict:
        """raw_features: a dict with FEATURE_COLS as keys (what the Fraud
        Scoring Agent's other tool calls -- get_account_info,
        get_tracking_status, etc. -- assemble). Returns risk_band,
        risk_score, and key_signals grounded in real SHAP values (Q34) --
        not the agent's own invented explanation."""
        row_df = pd.DataFrame([{col: raw_features[col] for col in FEATURE_COLS}])
        transformed_row = self.transformer.transform(row_df)
        proba = self.model.predict_proba(transformed_row)[0]
        predicted_idx = int(proba.argmax())

        top_features = top_contributing_features(
            self.explainer, transformed_row, self.feature_names, predicted_idx, top_n
        )
        return {
            "risk_band": CLASS_NAMES[predicted_idx],
            "risk_score": float(proba[predicted_idx]),
            "key_signals": [f["feature"] for f in top_features],
            "_attribution_detail": top_features,  # not part of the schema; useful for debugging/tests
        }


if __name__ == "__main__":
    print("Loading production model (v13, exported artifact)...")
    model = load_model()
    print("Rebuilding + refitting the ColumnTransformer (same split as training)...")
    transformer = build_and_fit_transformer()
    feature_names = list(transformer.get_feature_names_out())
    background = load_background(transformer)

    print("Building SHAP explainer (this is the potentially-slow part)...")
    start = time.time()
    explainer = build_explainer(model, background)
    build_time = time.time() - start
    print(f"Explainer built in {build_time:.2f}s")

    test_df = pd.read_csv(DATA_PATH)
    sample_rows = test_df[FEATURE_COLS + ["fraud_risk_label"]].sample(n=5, random_state=7)

    per_call_times = []
    for i, (_, row) in enumerate(sample_rows.iterrows(), 1):
        row_df = pd.DataFrame([row[FEATURE_COLS]])
        transformed_row = transformer.transform(row_df)
        proba = model.predict_proba(transformed_row)[0]
        predicted_idx = proba.argmax()
        predicted_class = CLASS_NAMES[predicted_idx]

        t0 = time.time()
        top_features = top_contributing_features(explainer, transformed_row, feature_names, predicted_idx)
        elapsed = time.time() - t0
        per_call_times.append(elapsed)

        print(f"\n--- Sample {i} (true label: {row['fraud_risk_label']}, predicted: {predicted_class}, p={proba[predicted_idx]:.3f}) ---")
        print(f"SHAP computation time: {elapsed:.3f}s")
        for feat in top_features:
            print(f"  {feat['feature']:<40} shap={feat['shap_value']:+.4f}")

    avg_time = sum(per_call_times) / len(per_call_times)
    print(f"\nAverage per-call SHAP time: {avg_time:.3f}s (explainer build, done once, was {build_time:.2f}s)")
