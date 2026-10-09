from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from config import IMAGE_DIR

def report(name, y_true, y_pred):
    print(f"--- {name} ---")
    print(f"R^2=  {metrics.r2_score(y_true, y_pred):.4f}")
    print(f"MSE=  {metrics.mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE= {metrics.root_mean_squared_error(y_true, y_pred):.4f}")

def  plot_SLR(X_test_SLR, Y_test_SLR, save=False):
    plt.figure(figsize=(6, 5))
    sns.regplot(
        x=X_test_SLR, y=Y_test_SLR,
        scatter_kws={"alpha": 0.25},
        line_kws={"color": "orange"}
    )
    plt.xlabel("Median Income")
    plt.ylabel("median House Value")
    plt.title("SLR: Reegression Line Fit")
    if save:
        IMAGE_DIR.mkdir(exist_ok=True)
        plt.savefig(IMAGE_DIR / "SLR_Regression_Line_Fit.png")
    plt.show()