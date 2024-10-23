import pandas as pd
from src.exception import CustomException
from src.analysis.handling_missing_values import MissingValueHandler,DropMissingValuesStrategy,FillMissingValuesStrategy

def handle_missing_values_step(df:pd.DataFrame,strategy:str ="mean"):
    """handle missing value using MissingValueHandler and specified strategy"""
    if strategy == 'drop':
        handler = MissingValueHandler(DropMissingValuesStrategy(axis=0))
    elif strategy in ['mean','median','mode','constant']:
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))
    else:
        raise CustomException(f"unsupported missing value handling {strategy}")

    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df