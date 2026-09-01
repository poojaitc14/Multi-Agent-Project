"""One-time build step (project-plan.md Q92): fits the ColumnTransformer
and samples the SHAP background distribution ONCE from
ml/data/synthetic_fraud_risk_dataset.csv, then pickles both so
ml/fraud_attribution.py's FraudAttributor never has to read the CSV or
call .fit() at runtime again -- every previous run (local, CI) refit this
same, deterministic transformer from scratch on every single server
startup, which is real, unnecessary repeated work every time the code
gets pushed or the server restarts.

Re-run this only when NUMERIC_COLS/BOOL_COLS/CATEGORICAL_COLS or the
training data itself changes -- the output is deterministic (same file,
same split, same random_state), so there's nothing to regenerate
otherwise.

Run with: uv run python ml/build_fraud_transformer.py
Writes: ml/fraud_transformer.pkl, ml/fraud_background.pkl
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_ML_DIR = Path(__file__).resolve().parent
DATA_PATH = str(_ML_DIR / "data" / "synthetic_fraud_risk_dataset.csv")
RANDOM_STATE = 42  # matches ml/model_training.ipynb exactly

NUMERIC_COLS = [
    "account_age_days", "total_orders_lifetime", "total_returns_lifetime", "claim_frequency_90d",
    "refund_amount_usd", "days_to_return", "customer_support_contacts_90d", "previous_dispute_count",
]
BOOL_COLS = ["address_match", "is_high_value_item", "photo_evidence_provided"]
CATEGORICAL_COLS = ["claim_category", "image_consistency"]
FEATURE_COLS = NUMERIC_COLS + BOOL_COLS + CATEGORICAL_COLS


def build_and_fit_transformer(df: pd.DataFrame) -> ColumnTransformer:
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


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    transformer = build_and_fit_transformer(df)
    print(f"Fitted ColumnTransformer, {len(transformer.get_feature_names_out())} output features")

    background_sample = df[FEATURE_COLS].sample(n=50, random_state=42)
    background_transformed = transformer.transform(background_sample)
    print(f"Transformed a {len(background_sample)}-row SHAP background sample")

    transformer_path = _ML_DIR / "fraud_transformer.pkl"
    background_path = _ML_DIR / "fraud_background.pkl"
    with open(transformer_path, "wb") as f:
        pickle.dump(transformer, f)
    with open(background_path, "wb") as f:
        pickle.dump(background_transformed, f)

    print(f"Wrote {transformer_path} ({transformer_path.stat().st_size} bytes)")
    print(f"Wrote {background_path} ({background_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
