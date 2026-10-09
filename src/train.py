from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from preprocessing import build_preprocessor
from config import CV_FOLDS, MODEL_DIR
import joblib

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

def train(pipeline, param_grid, X_train, Y_train):
    grid = GridSearchCV(pipeline, param_grid=param_grid,
                        cv=CV_FOLDS, scoring="r2", n_jobs=-1)
    grid.fit(X_train, Y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best R2:     {grid.best_score_:.4f}")
    return grid.best_estimator_

def save(model, name):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"{name}.pkl")
    print(f"saved {name}.pkl")
