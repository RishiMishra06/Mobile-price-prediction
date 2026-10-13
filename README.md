# Mobile Price Range Prediction

A machine learning web app that predicts the price range of a mobile phone (Low / Medium / High / Very High) from its specifications.

**Live demo:** <add your Streamlit link here>

## Overview
Built during my 6-week Machine Learning training at Winnovation. The app takes 20 phone specifications (RAM, battery, resolution, camera, etc.) and predicts the price range using a trained Logistic Regression model.

## Dataset
Mobile Price Classification dataset (2000 phones, 20 features, 4 price classes).

## Models compared
| Model | Accuracy |
|---|---|
| Logistic Regression | 97.5% |
| Random Forest | 89.25% |
| KNN | 53% |

Logistic Regression was selected as the final model.

## Key findings
- RAM is by far the strongest predictor of price range (correlation ~0.92).
- Battery power and screen resolution (px_height, px_width) also contribute.
- Bluetooth, WiFi, Dual SIM and Touch Screen have almost no effect.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Author
Rishi Mishra - B.Tech CSE, BPIT Rohini
