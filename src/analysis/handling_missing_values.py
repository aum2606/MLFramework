from src.logger import logging
from abc import ABC,abstractmethod
import pandas as pd

class HandlingMissingValuesStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Abstract method to handle missing values in dataframe

            Paramteres:
                df -> pd.dataframe [dataframe]

            Return:
                df -> pd.dataframe => return dataframe without missing values 
        """
        pass

class DropMissingValuesStrategy(HandlingMissingValuesStrategy):
    def __init__(self,axis=0,thresh=None) -> None:
        """
            Initializies the DropMissingValueStrategy with specific paramters

            Parameters:
                axis(int) -> 0 to drop row and 1 to drop column containing missing values
                thresh(int) -> threshold for non NA values [drops rows and columns with less thresholds]
            
            Return:
                None
        """
        self.axis = axis
        self.thresh = thresh
    
    def handle(self, df:pd.DataFrame)->pd.DataFrame:
        """
            drops row or columns with missing values based on axis and threshold

            Parameters:
                df -> pd.dataframe [dataframe]
            
            Returns:
                df -> pd.dataframe => return dataframe after dropping the missing values
        """
        logging.info(f"dropping the missing valus with axis = {self.axis} and threshold = {self.thresh} ")
        df_cleaned = df.dropna(axis=self.axis,thresh=self.thresh)
        logging.info("dropped the missing values")
        return df_cleaned


class FillMissingValuesStrategy(HandlingMissingValuesStrategy):
    def __init__(self,method='mean',fill_value=None):
        """
            initializing the FillMissingValueStrategy class with a specific methor or fill_value

            Parametes:
                method(str) -> method to fill the missing values ('mean','median','mode','constant')
                fill_value(any) -> constant value to fill missing values when method ='constant'
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) ->pd.DataFrame:
        """
            fills missing values using the specified method or constant values

            Parameters:
                df -> pd.dataframe [dataframe]
            
            Returns:
                df -> pd.dataframe => return dataframe after filling the missing values
        """
        logging.info(f"filling the missing values with method = {self.method} and fill_value = {self.fill_value}")

        df_cleaned = df.copy()
        if self.method == 'mean':
            numeric_columns = df_cleaned.select_dtypes(include='number').columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].mean())
        elif self.method == 'median':
            numeric_columns = df_cleaned.select_dtypes(include='number').columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].median())
        elif self.method == 'mode':
            for column in df_cleaned.columns:
                df_cleaned[column] = df_cleaned[column].fillna(df[column].mode().iloc[0],inplace=True)
        elif self.method == 'constant':
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.info(f"unknown method {self.method}. No missing values handled.")
        
        logging.info("missing values filled")
        return df_cleaned
    
class MissingValueHandler:
    def __init__(self, strategy: HandlingMissingValuesStrategy):
        """
        Initializes the MissingValueHandler with a specific missing value handling strategy.

        Parameters:
        strategy (MissingValueHandlingStrategy): The strategy to be used for handling missing values.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: HandlingMissingValuesStrategy):
        """
        Sets a new strategy for the MissingValueHandler.

        Parameters:
        strategy (MissingValueHandlingStrategy): The new strategy to be used for handling missing values.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the missing value handling using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)


if __name__ =="__main__":
    # Example usage
    # df = pd.read_csv('D:/coding/ml/machine learning framework/src/components/extracted_data/AmesHousing.csv')
    # missing_value_handler = MissingValueHandler(DropMissingValuesStrategy(axis=0,thresh=3))
    # df_cleaned = missing_value_handler.handle_missing_values(df)

    # missing_value_handler.set_strategy(FillMissingValuesStrategy(method='mean'))
    # df_filled = missing_value_handler.handle_missing_values(df)
    pass