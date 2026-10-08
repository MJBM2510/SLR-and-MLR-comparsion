# Paths
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "housing.csv"
MODEL_DIR = ROOT / "models"
IMAGE_DIR = ROOT / "image"


# Constants
TARGET = "median_house_value"
NUMERIC_COLS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income"
]
CATEGORICAL_COLS = ["ocean_proximity"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5