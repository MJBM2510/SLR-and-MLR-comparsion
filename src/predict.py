from train import load
import numpy as np

def predict(model_name, X):
    model = load(model_name)
    return model.predict(X)