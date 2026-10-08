from config import DATA_PATH, TARGET, RANDOM_STATE, TEST_SIZE
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    return pd.read_csv(filepath_or_buffer=DATA_PATH)

def split(df, features):
    X = df[features] if isinstance(features, list) else df.drop(columns=TARGET)
    Y = df[TARGET]
    return train_test_split(X, Y, random_state=RANDOM_STATE, test_size=TEST_SIZE)