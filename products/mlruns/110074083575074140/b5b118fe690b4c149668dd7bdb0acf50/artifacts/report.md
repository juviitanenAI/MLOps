# California Housing — Ridge Regression Report

> Auto-generated on run `b5b118fe690b4c149668dd7bdb0acf50`

---

## 1. Executive Summary

| Item | Value |
|------|-------|
| Dataset | `sklearn.datasets.fetch_california_housing()` |
| Samples | 20640 (train 16512, test 4128) |
| Features | 8: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude |
| Target | Median house value ($100 k) |
| Model | Ridge regression (α = 1.0) |
| Scaling | StandardScaler |

---

## 2. Metrics

| Metric | Value |
|--------|-------|
| **MSE**  | 0.5559 |
| **RMSE** | 0.7456 |
| **MAE**  | 0.5332 |
| **R²**   | 0.5758 |

---

## 3. Forecast Visualizations

### Actual vs Predicted
![Actual vs Predicted](images/end_to_end_project/actual_vs_predicted.png)

### Residuals
![Residuals](images/end_to_end_project/residuals.png)

---

## 4. Feature Importance

### Ridge Coefficients
![Coefficients](images/end_to_end_project/feature_importance_coefficients.png)

### Permutation Importance
![Permutation Importance](images/end_to_end_project/feature_importance_permutation.png)

### SHAP Summary
![SHAP Summary](images/end_to_end_project/shap_summary.png)

---

## 5. Conclusions & Next Steps

- **R² = 0.5758** — the linear model captures ~58% of variance. Reasonable for a
  simple linear model, but non-linear models (e.g. Random Forest, Gradient Boosting) would
  likely improve this.
- The residuals plot shows heteroscedasticity at higher predicted values, suggesting the
  relationship is not purely linear.
- **Top features** (by permutation importance and SHAP) indicate which housing
  characteristics drive price the most — useful for targeted data collection and
  feature engineering.
- **Potential improvements**: try ensemble models, add polynomial / interaction features,
  hyperparameter tuning via cross-validation, and outlier handling.
