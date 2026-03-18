# MLOps — California Housing with MLflow

Train a Ridge regression model on the California Housing dataset, log everything with MLflow, and generate a comprehensive report with metrics, visualizations, and feature importance.

## Quick Start

```bash
# Clone
git clone https://github.com/juviitanenAI/MLOps.git
cd MLOps

# Virtual environment
python -m venv env
source env/bin/activate        # macOS / Linux
# .\env\Scripts\activate       # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the pipeline
python calhousemlflow.py
```

## What the Script Does

1. Loads `sklearn.datasets.fetch_california_housing()`
2. Splits data 80/20 (stratified by random state 42)
3. Scales features with `StandardScaler`
4. Trains a **Ridge regression** (α = 1.0)
5. Logs params, metrics (MSE, RMSE, MAE, R²), and the model to **MLflow**
6. Loads the model back via MLflow and verifies predictions
7. Generates visualizations:
   - Actual vs Predicted scatter
   - Residuals plot
   - Ridge coefficient magnitudes
   - Permutation feature importance
   - SHAP summary
8. Writes `products/report.md` with all results

## View MLflow UI

```bash
mlflow ui --backend-store-uri file://$(pwd)/products/mlruns
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Output Structure

```
products/
├── images/end_to_end_project/   # All generated plots (PNG)
├── mlruns/                      # MLflow tracking store
└── report.md                    # Auto-generated report
```
