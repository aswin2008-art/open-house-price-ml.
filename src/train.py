"""Train and evaluate a simple house-price regression model.

The bundled data is synthetic and intended for learning only.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing_sample.csv"
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years"]
TARGET = "price_usd"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data.empty:
        raise ValueError("Dataset has no rows.")
    data = data.dropna(subset=[TARGET])
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Evaluation on held-out synthetic sample data")
    print(f"Mean absolute error (MAE): ${mean_absolute_error(y_test, predictions):,.0f}")
    print(f"R² score: {r2_score(y_test, predictions):.3f}")
    print("Reminder: these metrics do not indicate real-world housing accuracy.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
