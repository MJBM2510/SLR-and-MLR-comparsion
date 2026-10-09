from config import TARGET, NUMERIC_COLS, CATEGORICAL_COLS
from data import *
from evaluate import *
from train import *
import pandas as pd
from sklearn import metrics

SLR_PARAM_GRID = {
    "imputer__strategy": ["mean", "median"],
    "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}

MLR_PARAM_GRID = MLR_grid_param = {
    "preprocessor__numerical__imputer__strategy": ["mean", "median"],
    "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}

if __name__ == "__main__":
    df = load_data()
    
    X_train_SLR, X_test_SLR, Y_train_SLR, Y_test_SLR = split(df, ["median_income"])
    SLR = train(build_SLR_pipeline(), SLR_PARAM_GRID,
                X_train_SLR, Y_train_SLR)
    SLR_predictions = SLR.predict(X_test_SLR)
    report(SLR, Y_test_SLR, SLR_predictions)
    save(SLR, "SLR_model")
    
    X_train_MLR, X_test_MLR, Y_train_MLR, Y_test_MLR = split(df, None)
    MLR = train(build_MLR_pipeline(), MLR_PARAM_GRID,
                X_train_MLR, Y_train_MLR)
    MLR_predictions = MLR.predict(X_test_MLR)
    report("MLR", Y_test_MLR, MLR_predictions)
    save(MLR, "MLR_model")
    
    plot_SLR(X_test_SLR, Y_test_SLR, True)
    plot_MLR(Y_test_MLR, MLR_predictions, True)
    plot_comparison(
        metrics.r2_score(Y_test_SLR, SLR_predictions),
        metrics.r2_score(Y_test_MLR, MLR_predictions),
        save=True
    )