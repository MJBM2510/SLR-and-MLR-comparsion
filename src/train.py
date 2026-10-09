from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from preprocessing import build_preprocessor

def build_SLR_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="medain")),
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])

def build_MLR_pipeline():
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", Ridge())
    ])