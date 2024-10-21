import pandas as pd
from src.exception import CustomException
from src.components.data_preprocessing import(
    PreProcessor,
    LogTransformation,
    MinMaxScaling,
    StandardScaling,
    OneHotEncoding,
)

def data_preprocessing_step(df:pd.DataFrame,strategy:str='log',featrues:list=None)->pd.DataFrame:
    """Perform data pre-processing using various data pre-processors based on selected strategy"""
    if featrues is None:
        featrues=[]

    if strategy == 'log':
        data_preprocessor = PreProcessor(LogTransformation(features=featrues))
    elif strategy == 'standard_scaling':
        data_preprocessor = PreProcessor(StandardScaling(features=featrues))
    elif strategy == 'minmax_scaling':
        data_preprocessor = PreProcessor(MinMaxScaling(features=featrues))
    elif strategy == 'onehot_encoding':
        data_preprocessor = PreProcessor(OneHotEncoding(features=featrues))
    else:
        raise CustomException(f"Unsupported data preprocessing strategy: {strategy}")