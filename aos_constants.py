from typing import List, TypeVar
import pandas as pd

T = TypeVar('T')
R = TypeVar('R')

RESULTS: str = "results"
STUDY_1: str = f"{RESULTS}/study_1"
STUDY_2: str = f"{RESULTS}/study_2"

def drop_not_columns(df: pd.DataFrame, columns: List[str]):
    return df.drop(columns=[e for e in df.columns if e not in columns])
