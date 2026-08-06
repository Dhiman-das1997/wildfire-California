This repository presents a leakage-free machine learning framework for ** wildfire prediction** across ten wildfire-prone counties in California using daily meteorological data.

The pipeline includes:

- Data preprocessing and county-level panel construction
- Causal feature engineering using lagged, rolling, seasonal, and weather-derived variables
- Leakage-free temporal cross-validation with expanding-window outer folds and forward-chaining inner folds
- Training of three predictive models:
  - Logistic Regression
  - XGBoost
  - Long Short-Term Memory (LSTM)
- Probability calibration using Platt scaling
- County-specific threshold optimization for operational warning generation
- Comprehensive performance evaluation using out-of-fold predictions
- Automatic generation of publication-quality figures and performance summaries

The framework is designed for reproducible wildfire forecasting while ensuring strict prevention of temporal data leakage throughout model development and evaluation.
