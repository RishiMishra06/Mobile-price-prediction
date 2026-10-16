# Mobile Price Predictor

![tests](https://github.com/<your-username>/mobile-price-prediction/actions/workflows/tests.yml/badge.svg)

A machine learning web app that estimates a smartphone's market price in India, tells you whether an offer is a good deal, and recommends the best phones for your budget. Built with Python, Scikit-learn and Streamlit.

**Live demo:** <https://mobile-price-prediction-rm160206.streamlit.app/>

<!-- Add 2-3 screenshots or a short GIF of the app here, for example:
![Predict tab](docs/predict.png) -->

## Features
| Area | What it does |
|---|---|
| **Predict** | Price in ₹ from RAM, storage, battery, charging, cameras and AnTuTu score, with an 80% prediction interval, a Shapley explanation of every spec's effect, and a PDF report |
| **Deal check** | Enter the price you are offered; get a verdict (great deal to overpriced) on a visual gauge, better-value alternatives, and an EMI calculator |
| **Phones** | Budget + priority ranking with filters (brand, 5G, RAM, storage, battery), side-by-side comparison with PDF export, similar phones, value-for-money chart, wishlist, editable catalog |
| **Insights / Models** | Model card, Shapley importance, actual vs predicted plot, cross-validation, live GridSearchCV tuning for the classifiers |
| **Batch** | Upload a CSV to estimate prices or classify segments in bulk |

Every spec has a slider and a number box, with ranges that match phones sold today.

## Two models
| Model | Data | Purpose |
|---|---|---|
| Gradient Boosting regression | `phones.csv` (54 phones sold in India, approx. Oct 2026 prices) | Price in ₹ for modern phones |
| Logistic Regression classifier | Kaggle Mobile Price Classification (2000 phones, 4 classes) | Price segment, from my Winnovation training report |

The Kaggle dataset has only price classes (0-3) and older phones (RAM up to ~4 GB), so it cannot price a modern phone in ₹. The India catalog model does; the classifier is kept to show the full project.

## How the market model was built
- **Tuning:** GridSearchCV over trees, depth, learning rate, leaf size and subsampling
- **Evaluation:** nested cross-validation (6-fold x 2 repeats, grid search inside each fold), because with 54 phones a plain train/test split is unreliable
- **Result:** R² ≈ 0.75, mean absolute error ≈ ₹7,100 (≈ 23% average error). Tuning improved on the untuned model only slightly (R² 0.74, MAE ≈ ₹7,300): the data is the limit, not the algorithm
- **Prediction interval:** conformal prediction from out-of-sample residuals (80% target, 83% observed). Quantile regression was tried first but covered only 63%, so it was not used
- **Explanations:** exact Shapley values (all 2^7 feature coalitions), rescaled to ₹
- **Persistence:** the trained model is saved with `joblib` (`market_model.joblib`); the app retrains automatically if the catalog or scikit-learn version differs

Classification models from the report (80/20 split): Logistic Regression 97.5%, Random Forest 89.25%, KNN 53% (the app can re-tune them live).

## Limitations
- The catalog is hand-collected from public listings; prices vary by variant and seller, and sources sometimes disagree
- 54 phones is small, so estimates are indicative (about ±₹7,000), and brand premium (e.g. Apple) is not captured
- The 5G flag is derived from the chipset/model name
- The phone ranking is a transparent spec-based score, not a review; megapixels do not measure camera quality

## Project structure
```
app.py              Streamlit UI
core.py             Models, scoring, Shapley, EMI, deal logic (no Streamlit, unit tested)
pdf_report.py       PDF reports (ReportLab)
train_models.py     GridSearchCV + nested CV, saves market_model.joblib and model_card.json
phones.csv          Phone catalog
train.csv           Kaggle dataset used by the classification model
test_core.py        24 unit tests
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Update the catalog, then retrain and test:
```
python train_models.py
pip install pytest && python -m pytest -q
```

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Altair, ReportLab, Streamlit, pytest, GitHub Actions

## Author
Rishi Mishra - B.Tech CSE, BPIT Rohini
