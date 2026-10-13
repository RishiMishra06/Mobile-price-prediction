# Mobile Price Predictor

A machine learning web app that predicts a mobile phone's price segment (Low to Very High) and an indicative price in INR from 20 specifications, using Logistic Regression (97.5% accuracy). Built with Python, Scikit-learn and Streamlit.

**Live demo:** <add your Streamlit link here>

## Features
- Predict: sliders and toggles for 20 specs, one-click Budget / Mid-range / Flagship presets
- Estimated price in INR with confidence and per-class probabilities
- "Why this prediction?" chart showing which features drove the result
- Phones: budget + priority based recommendations and side-by-side comparison of phones sold in India (hand-collected catalog in phones.csv, approximate Oct 2026 prices)
- Phones available near the predicted price are suggested right after each prediction
- Batch prediction: upload a CSV, download predictions
- Insights: class distribution, average specs per class, correlations, feature importance
- Models: live comparison of Logistic Regression, Random Forest and KNN with confusion matrix and classification report
- Prediction history with CSV download

## Note on the price estimate
The dataset contains price classes (0-3), not actual prices. The INR figure maps each class to an editable price band and weights it by the model's class probabilities, so it is an indicative estimate rather than a true regression output.

## Note on phone recommendations
The catalog (phones.csv) was collected manually from public listings and prices vary by variant and seller. The score is spec-based (RAM, storage, battery, charging, camera MP, display size, price) and separate from the ML model.

## Dataset
Mobile Price Classification dataset (2000 phones, 20 features, 4 price classes).

## Models compared
| Model | Accuracy |
|---|---|
| Logistic Regression | 97.5% |
| Random Forest | 89.25% |
| KNN | 53% |

## Key findings
- RAM is by far the strongest predictor of price range (correlation ~0.92).
- Battery power and screen resolution also contribute.
- Bluetooth, WiFi, Dual SIM and Touch Screen have almost no effect.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Streamlit

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Author
Rishi Mishra - B.Tech CSE, BPIT Rohini
