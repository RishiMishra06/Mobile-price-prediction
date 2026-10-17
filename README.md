# SpecWorth

*Know what your phone is worth.*

[![Live demo](https://img.shields.io/badge/Live%20demo-specworth.streamlit.app-FF4B4B?logo=streamlit&logoColor=white)](https://specworth.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Gradient%20Boosting-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

SpecWorth is a machine learning web app that estimates a smartphone's market price in India from its specifications, tells you whether an offer is a good deal, and recommends the best phones for your budget.

**Try it:** [specworth.streamlit.app](https://specworth.streamlit.app/)

<p align="center">
  <img src="docs/predict.png" width="48%" alt="Predict tab: a slider and number box for every specification">
  <img src="docs/predict-result.png" width="48%" alt="Estimated price with range, Shapley chart and phones around that price">
</p>
<p align="center">
  <img src="docs/deal-check.png" width="48%" alt="Deal check with verdict, price gauge and better-value phones">
  <img src="docs/phones.png" width="48%" alt="Phone recommendations for a budget up to 1.5 lakh rupees">
</p>
<p align="center">
  <img src="docs/emi-calculator.png" width="48%" alt="EMI calculator with a tenure comparison table">
</p>

## Contents
[Features](#features) · [Two models](#two-models) · [Model performance](#model-performance) · [How it works](#how-it-works) · [Run locally](#run-locally) · [Project structure](#project-structure) · [Limitations](#limitations) · [Author](#author)

## Features

| Area | What it does |
|---|---|
| **Predict** | Price in ₹ from RAM, storage, battery, charging, cameras and AnTuTu score, with an 80% prediction interval, a Shapley explanation of every spec's effect, and a PDF report. Every spec has a slider and a number box with ranges that match phones sold today, and RAM and storage move in the sizes real phones come in |
| **Deal check** | Enter the price you are offered. Catalog phones are compared with their listed price, and offers far below it are flagged as *too good to be true*; your own specs are compared with the model's estimate. Visual gauge, better-value alternatives and an EMI calculator |
| **Phones** | Budget (₹5,000 to ₹10 lakh) and priority ranking with filters (brand, 5G, RAM, storage, battery), side-by-side comparison with PDF export, similar phones, value-for-money chart, wishlist, editable catalog |
| **Insights & Models** | Model card, Shapley importance, actual vs predicted plot, cross-validation, live GridSearchCV tuning for the classifiers |
| **Batch** | Upload a CSV to estimate prices or classify price segments in bulk |

## Two models

| Model | Data | Purpose |
|---|---|---|
| Gradient Boosting regression | `phones.csv`: 63 phones sold in India, ₹7,989 to ₹1,94,999 (approx. Oct 2026 prices). 61 are used for training; the 2 foldables are listed but excluded because the fold premium is not explained by specs | Price in ₹ for modern phones |
| Logistic Regression classifier | Kaggle Mobile Price Classification: 2,000 phones, 4 price classes | Price segment, from my Winnovation ML training project |

The Kaggle dataset only has price classes (0-3) and older phones (RAM up to about 4 GB), so it cannot price a modern phone in ₹. The India catalog model does that, and the classifier is kept to show the full project.

## Model performance

Regression model, 61 phones:

| Evaluation | R² | Mean absolute error | Average error |
|---|---|---|---|
| Nested cross-validation (6-fold x 2, grid search inside each fold) | 0.81 | about ₹12,000 | about 26% |
| Leave-one-out score of the final tuned model (shown in the app, which retrains on first load, so the last digit can vary) | 0.87 to 0.88 | about ₹9,000 to ₹9,500 | about 21% to 22% |
| Untuned baseline (leave-one-out) | 0.84 | about ₹9,900 | - |

Nested CV is the honest estimate: the hyperparameters were tuned on the same phones, so the leave-one-out score of the tuned model is optimistic. Tuning brought no clear gain over the untuned baseline. With about 60 phones, the data is the limit rather than the algorithm.

**The error grows with the price** (nested CV, out-of-fold predictions):

| Price band | Phones | Mean absolute error | Average error |
|---|---|---|---|
| Under ₹15,000 | 14 | about ₹2,600 | 22% |
| ₹15,000 to ₹30,000 | 25 | about ₹5,300 | 22% |
| ₹30,000 to ₹60,000 | 10 | about ₹10,200 | 26% |
| Above ₹60,000 | 12 | about ₹35,200 | 33% |

Budget and mid-range phones are estimated well. Above ₹60,000 there are few phones and brand value matters, so the estimate is much rougher.

**Prediction interval:** conformal prediction from out-of-sample residuals (80% target, 82% observed). Quantile regression was tried first but covered only 49% of phones, so it was not used.

**Classification models (80/20 split):** Logistic Regression 97.5%, Random Forest 89.25%, KNN 53%. The app can re-tune all three live.

## How it works
- **Features:** RAM, storage, battery, charging speed, rear and front camera, AnTuTu benchmark
- **Target:** log of the price, so errors are relative rather than absolute
- **Tuning:** GridSearchCV over trees, depth, learning rate, leaf size and subsampling
- **Explanations:** exact Shapley values over all 2^7 feature coalitions, rescaled to ₹ so they add up to the estimate
- **Brand value:** the model sees specs only, so Apple and Samsung flagships come out cheaper than their real price. That is why the deal checker uses the listed price for phones in the catalog
- **Persistence:** the trained model is saved with `joblib` (`market_model.joblib`); the app retrains automatically if the catalog or library version changes
- **Phone ranking:** a transparent spec-based score (AnTuTu, RAM, storage, battery, charging, camera MP, display size, price) weighted by the priorities you choose

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Needs `train.csv` (Kaggle Mobile Price Classification) and `phones.csv` in the same folder.

To update the catalog, retrain, and run the tests:

```bash
python train_models.py
pip install pytest
python -m pytest -q
```

## Project structure

```
app.py              Streamlit interface
core.py             Models, scoring, Shapley values, EMI and deal logic (no Streamlit, unit tested)
pdf_report.py       PDF reports (ReportLab)
train_models.py     GridSearchCV + nested CV; saves market_model.joblib and model_card.json
phones.csv          Phone catalog
train.csv           Kaggle dataset used by the classification model
test_core.py        26 unit tests
docs/               Screenshots
```

## Limitations
- The catalog is hand-collected from public listings. Prices vary by variant and seller, and sources sometimes disagree
- About 60 phones is small, so estimates are indicative, and brand premium (for example Apple) and the fold premium are not captured
- The 5G flag is derived from the chipset and model name
- The phone ranking is a spec-based score, not a review. Megapixels do not measure camera quality

## Tech stack
Python · Pandas · NumPy · scikit-learn · Altair · ReportLab · Streamlit · pytest · GitHub Actions

## Author
**Rishi Mishra**

## License
MIT, see [LICENSE](LICENSE).
