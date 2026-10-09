from sklearn import metrics

def report(name, y_true, y_pred):
    print(f"--- {name} ---")
    print(f"R^2=  {metrics.r2_score(y_true, y_pred):.4f}")
    print(f"MSE=  {metrics.mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE= {metrics.root_mean_squared_error(y_true, y_pred):.4f}")