# ============================================
#   MLflow Setup (MUST be FIRST in file)
# ============================================
import os
from pathlib import Path

# Create a dedicated directory for all generated products (logs, images, datasets, etc.)
PRODUCTS_DIR = os.path.abspath("products")
os.makedirs(PRODUCTS_DIR, exist_ok=True)

# Remove any environment overrides BEFORE importing mlflow
os.environ.pop("MLFLOW_TRACKING_URI", None)
os.environ.pop("MLFLOW_ARTIFACT_URI", None)

# Now import MLflow safely
import mlflow
import mlflow.sklearn

# Force MLflow to use your local file store
mlruns_dir = os.path.join(PRODUCTS_DIR, "mlruns")
mlruns_uri = Path(mlruns_dir).as_uri()

mlflow.set_tracking_uri(mlruns_uri)

# Debug + safety check
uri_now = mlflow.get_tracking_uri()
print(f"[DEBUG] MLflow tracking URI at script start: {uri_now}")
assert uri_now == mlruns_uri, f"Unexpected MLflow URI: {uri_now}"


# ============================================
#   Standard imports
# ============================================
import sys
assert sys.version_info >= (3, 7)

from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import shap


# ============================================
#   Matplotlib Setup
# ============================================
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PRODUCTS_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """Save figures and log them to MLflow."""
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, dpi=resolution)
    mlflow.log_artifact(path)
    print(f"  [SAVED] {path}")


# ============================================
#   Load Data — sklearn California Housing
# ============================================
print("\n=== Loading California Housing dataset ===")
SKLEARN_DATA_HOME = os.path.join(PRODUCTS_DIR, "sklearn_data")
os.makedirs(SKLEARN_DATA_HOME, exist_ok=True)
housing = fetch_california_housing(as_frame=True, data_home=SKLEARN_DATA_HOME)
X = housing.data          # DataFrame with 8 features
y = housing.target        # Series: MedHouseVal (in $100k)
feature_names = list(X.columns)

print(f"  Features: {feature_names}")
print(f"  Samples : {len(X)}")


# ============================================
#   Train / Test Split
# ============================================
TEST_SIZE = 0.2
RANDOM_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")


# ============================================
#   Feature Scaling
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================
#   MLflow Experiment
# ============================================
EXPERIMENT_NAME = "California_Housing_Ridge"
mlflow.set_experiment(EXPERIMENT_NAME)

ALPHA = 1.0

with mlflow.start_run(run_name="Ridge_Regression") as run:
    run_id = run.info.run_id
    print(f"\n=== MLflow run started: {run_id} ===")

    # ------------------------------------------
    # Log params
    # ------------------------------------------
    mlflow.log_param("model_type", "Ridge")
    mlflow.log_param("alpha", ALPHA)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("num_features", len(feature_names))
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))

    # ------------------------------------------
    # Train
    # ------------------------------------------
    print("\n=== Training Ridge regression ===")
    model = Ridge(alpha=ALPHA, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------
    # Evaluate on test set
    # ------------------------------------------
    y_pred = model.predict(X_test_scaled)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"  MSE  = {mse:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  R²   = {r2:.4f}")

    mlflow.log_metrics({"mse": mse, "rmse": rmse, "mae": mae, "r2": r2})

    # ------------------------------------------
    # Log model
    # ------------------------------------------
    print("\n=== Logging model to MLflow ===")
    mlflow.sklearn.log_model(model, "ridge_model")

    # ------------------------------------------
    # 1) Actual vs Predicted scatter
    # ------------------------------------------
    print("\n=== Generating visualizations ===")

    fig, ax = plt.subplots(figsize=(8, 8))
    residuals = y_test.values - y_pred
    scatter = ax.scatter(y_test, y_pred, c=np.abs(residuals), cmap="coolwarm",
                         alpha=0.5, s=10, edgecolors="none")
    plt.colorbar(scatter, ax=ax, label="| Residual |")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="Ideal")
    ax.set_xlabel("Actual MedHouseVal ($100k)")
    ax.set_ylabel("Predicted MedHouseVal ($100k)")
    ax.set_title("Actual vs Predicted — Ridge Regression")
    ax.legend()
    save_fig("actual_vs_predicted")
    plt.close()

    # ------------------------------------------
    # 2) Residuals plot
    # ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.4, s=10, edgecolors="none", color="steelblue")
    ax.axhline(y=0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Predicted MedHouseVal ($100k)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals Plot — Ridge Regression")
    save_fig("residuals")
    plt.close()

    # ------------------------------------------
    # 3) Feature importance — Ridge coefficients
    # ------------------------------------------
    coef_abs = np.abs(model.coef_)
    sorted_idx = np.argsort(coef_abs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(sorted_idx)), coef_abs[sorted_idx], color="teal")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("| Coefficient | (scaled features)")
    ax.set_title("Ridge Coefficient Magnitudes")
    save_fig("feature_importance_coefficients")
    plt.close()

    # ------------------------------------------
    # 4) Permutation importance
    # ------------------------------------------
    print("  Computing permutation importance …")
    perm = permutation_importance(model, X_test_scaled, y_test,
                                  n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    perm_sorted_idx = perm.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(perm.importances[perm_sorted_idx].T, vert=False,
               tick_labels=[feature_names[i] for i in perm_sorted_idx])
    ax.set_xlabel("Decrease in R² score")
    ax.set_title("Permutation Feature Importance")
    save_fig("feature_importance_permutation")
    plt.close()

    # ------------------------------------------
    # 5) SHAP summary
    # ------------------------------------------
    print("  Computing SHAP values …")
    # Use a background sample for speed
    bg = X_train_scaled[:100]
    explainer = shap.Explainer(model.predict, bg, feature_names=feature_names)
    shap_values = explainer(X_test_scaled[:500])

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, show=False)
    plt.title("SHAP Summary — Ridge Regression")
    save_fig("shap_summary")
    plt.close()

    # ------------------------------------------
    # Load model back from MLflow & predict
    # ------------------------------------------
    print("\n=== Loading model back from MLflow ===")
    model_uri = f"runs:/{run_id}/ridge_model"
    loaded_model = mlflow.sklearn.load_model(model_uri)

    y_pred_loaded = loaded_model.predict(X_test_scaled)
    assert np.allclose(y_pred, y_pred_loaded), "Loaded model predictions != original!"
    print("  ✅ Loaded model predictions match original.")

    # Print sample predictions
    sample_df = pd.DataFrame({
        "Actual":    y_test.values[:10],
        "Predicted": y_pred_loaded[:10],
        "Residual":  y_test.values[:10] - y_pred_loaded[:10]
    })
    print("\n  Sample predictions (first 10):")
    print(sample_df.to_string(index=False))

    # ------------------------------------------
    # Generate report.md
    # ------------------------------------------
    print("\n=== Generating report ===")

    # Build image links (relative to products/)
    img_rel = f"images/{CHAPTER_ID}"

    report = f"""# California Housing — Ridge Regression Report

> Auto-generated on run `{run_id}`

---

## 1. Executive Summary

| Item | Value |
|------|-------|
| Dataset | `sklearn.datasets.fetch_california_housing()` |
| Samples | {len(X)} (train {len(X_train)}, test {len(X_test)}) |
| Features | {len(feature_names)}: {', '.join(feature_names)} |
| Target | Median house value ($100 k) |
| Model | Ridge regression (α = {ALPHA}) |
| Scaling | StandardScaler |

---

## 2. Metrics

| Metric | Value |
|--------|-------|
| **MSE**  | {mse:.4f} |
| **RMSE** | {rmse:.4f} |
| **MAE**  | {mae:.4f} |
| **R²**   | {r2:.4f} |

---

## 3. Forecast Visualizations

### Actual vs Predicted
![Actual vs Predicted]({img_rel}/actual_vs_predicted.png)

### Residuals
![Residuals]({img_rel}/residuals.png)

---

## 4. Feature Importance

### Ridge Coefficients
![Coefficients]({img_rel}/feature_importance_coefficients.png)

### Permutation Importance
![Permutation Importance]({img_rel}/feature_importance_permutation.png)

### SHAP Summary
![SHAP Summary]({img_rel}/shap_summary.png)

---

## 5. Conclusions & Next Steps

- **R² = {r2:.4f}** — the linear model captures ~{r2*100:.0f}% of variance. Reasonable for a
  simple linear model, but non-linear models (e.g. Random Forest, Gradient Boosting) would
  likely improve this.
- The residuals plot shows heteroscedasticity at higher predicted values, suggesting the
  relationship is not purely linear.
- **Top features** (by permutation importance and SHAP) indicate which housing
  characteristics drive price the most — useful for targeted data collection and
  feature engineering.
- **Potential improvements**: try ensemble models, add polynomial / interaction features,
  hyperparameter tuning via cross-validation, and outlier handling.
"""

    report_path = os.path.join(PRODUCTS_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)
    print(f"  [SAVED] {report_path}")


# ============================================
#   Done
# ============================================
print("\n" + "=" * 50)
print("MLflow run completed. View results using:")
print(f"  mlflow ui --backend-store-uri {mlruns_uri}")
print(f"  Report: {os.path.join(PRODUCTS_DIR, 'report.md')}")
print("=" * 50)