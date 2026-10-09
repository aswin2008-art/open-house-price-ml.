# Open House Price ML

A beginner-friendly, reproducible machine-learning project that demonstrates a house-price regression workflow with Python and scikit-learn.

> **Important:** `data/housing_sample.csv` contains **synthetic teaching data**, not real property listings or reliable market prices. Do not use the model for real-world buying, selling, lending, or valuation decisions. Replace it with a properly licensed, documented real dataset before drawing real-world conclusions.

## What this project demonstrates

- Loading and validating tabular data
- Splitting data into training and test sets
- Training a scikit-learn regression pipeline
- Evaluating predictions with MAE and R²
- Saving a trained model
- Making a prediction for a sample property

## Features used

- `area_sqft`: floor area in square feet
- `bedrooms`: number of bedrooms
- `bathrooms`: number of bathrooms
- `age_years`: approximate property age in years
- `price_usd`: target value in the **synthetic example dataset only**

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- Windows: `.venv\\Scripts\\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train and evaluate

From the project root:

```bash
python -m src.train
```

The script prints MAE and R² for the held-out test set and saves the trained model to `models/house_price_model.joblib`.

## Make a sample prediction

After training:

```bash
python -m src.predict --area 1500 --bedrooms 3 --bathrooms 2 --age 10
```

The output is only a demonstration based on synthetic data.

## Data and responsible use

The included dataset is synthetic and deliberately small, so its metrics are not evidence of real-world accuracy. If you replace it with a real dataset:

1. Confirm that its license allows your intended use.
2. Document the source, collection date, geography, units, and known limitations.
3. Check for missing values, outliers, data leakage, and bias.
4. Evaluate on data representative of the intended use.
5. Avoid presenting predictions as appraisals or financial advice.

## Contributing

Issues and pull requests are welcome. Please include steps to reproduce bugs and tests or evaluation evidence for changes. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

- [ ] Add a documented, appropriately licensed real-world dataset
- [ ] Add unit tests and continuous integration
- [ ] Compare linear regression with baseline models
- [ ] Add plots for residuals and model evaluation
- [ ] Improve data validation and error messages

## License

Released under the MIT License. See [LICENSE](LICENSE).
