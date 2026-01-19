import pandas as pd
from src.config import DATA_PATH

def extract_transactions() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=",", decimal=".")
    return df