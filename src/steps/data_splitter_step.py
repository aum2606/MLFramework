from typing import Tuple
import pandas as pd
from src.components.data_splitter import (
    DataSplitter,
    SimpleTrainTestSplitStrategy
)

def data_splitter_step(df: pd.DataFrame,target_column:str)->Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    """Split the data into training and testing sets."""
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train,X_test,y_train,y_test = splitter.split(df=df,target_column=target_column)
    return X_train,X_test,y_train,y_test


