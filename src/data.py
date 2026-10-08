from config import DATA_PATH
import pandas as pd

def load_data():
    return pd.read_csv(filepath_or_buffer=DATA_PATH)