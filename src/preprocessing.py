from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from config import NUMERIC_COLS, CATEGORICAL_COLS

def build_preprocessor():
    numerical_processor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    return ColumnTransformer(transformers=[
        ("numerical", numerical_processor, NUMERIC_COLS),
        ("categorical", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_COLS)
    ])