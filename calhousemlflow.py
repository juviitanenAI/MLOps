# ============================================
#   MLflow Setup (MUST be FIRST in file)
# ============================================
import os

# Remove any environment overrides BEFORE importing mlflow
os.environ.pop("MLFLOW_TRACKING_URI", None)
os.environ.pop("MLFLOW_ARTIFACT_URI", None)

# Now import MLflow safely
import mlflow
import mlflow.sklearn

# Force MLflow to use your local file store (NOT SQLite, NOT OneDrive!)
mlflow.set_tracking_uri("file:///C:/mlruns")

# Debug + safety check
uri_now = mlflow.get_tracking_uri()
print(f"[DEBUG] MLflow tracking URI at script start: {uri_now}")
assert uri_now == "file:///C:/mlruns", f"Unexpected MLflow URI: {uri_now}"


# ============================================
#   Standard imports
# ============================================
import sys
assert sys.version_info >= (3, 7)

from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tarfile
import urllib.request
import pandas as pd


# ============================================
#   Matplotlib Setup
# ============================================
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """Save figures and log them to MLflow."""
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, dpi=resolution)
    mlflow.log_artifact(path)


# ============================================
#   Download Housing Dataset
# ============================================
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# ============================================
#   MLflow Experiment
# ============================================
mlflow.set_experiment("Housing_Project_Experiment")

with mlflow.start_run(run_name="Data_Preparation"):

    mlflow.log_param("dataset_url", HOUSING_URL)

    # Fetch + load
    fetch_housing_data()
    housing = load_housing_data()

    # Dataset size
    mlflow.log_param("num_rows", housing.shape[0])
    mlflow.log_param("num_columns", housing.shape[1])

    # ----------------------------------------
    # Train-test split
    # ----------------------------------------
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    mlflow.log_param("train_size", len(train_set))
    mlflow.log_param("test_size", len(test_set))

    # ----------------------------------------
    # Income category
    # ----------------------------------------
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    # Plot and log
    housing["income_cat"].hist()
    save_fig("income_category_hist")
    plt.close()

    # Log counts
    income_counts = housing["income_cat"].value_counts()
    for label, count in income_counts.items():
        mlflow.log_metric(f"income_cat_{label}", int(count))

    # ----------------------------------------
    # Stratified split
    # ----------------------------------------
    from sklearn.model_selection import StratifiedShuffleSplit

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    mlflow.log_param("strat_train_size", len(strat_train_set))
    mlflow.log_param("strat_test_size", len(strat_test_set))

    # Save artifacts
    strat_train_set.head().to_csv("strat_train_sample.csv", index=False)
    mlflow.log_artifact("strat_train_sample.csv")

    strat_test_set.head().to_csv("strat_test_sample.csv", index=False)
    mlflow.log_artifact("strat_test_sample.csv")


print("MLflow run completed. View results using:")
print("  mlflow ui --backend-store-uri file:///C:/mlruns")