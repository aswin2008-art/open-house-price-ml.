"""Predict a demonstration price from user-supplied property features."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", type=float, required=True, help="Area in square feet")
    parser.add_argument("--bedrooms", type=int, required=True)
    parser.add_argument("--bathrooms", type=float, required=True)
    parser.add_argument("--age", type=float, required=True, help="Property age in years")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run `python -m src.train` first.")

    bundle = joblib.load(MODEL_PATH)
    features = bundle["features"]
    sample = pd.DataFrame([{
        "area_sqft": args.area,
        "bedrooms": args.bedrooms,
        "bathrooms": args.bathrooms,
        "age_years": args.age,
    }], columns=features)
    prediction = bundle["model"].predict(sample)[0]
    print(f"Demonstration predicted price: ${prediction:,.0f}")
    print("Synthetic-data demo only; not a real-world valuation.")


if __name__ == "__main__":
    main()
