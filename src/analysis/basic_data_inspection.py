from abc import ABC, abstractmethod
import pandas as pd


class DataInspectionStrategy(ABC):
    @abstractmethod
    def  inspect(self, df:pd.DataFrame):
        pass


class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df:pd.DataFrame):
        """
            inspecting and printing the data types and non null counts of the data frame columns.

            parameters:
                df -> pd.dataframe [dataframe]

            Returns:
            None -> printing data information on data types and non null counts
        """
        print("data types and non-null counts: ")
        print(df.info())


class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df:pd.DataFrame):
        """
            print summary statistics for the numerical and categorical features

            parameters:
                df-> pd.dataframe [dataframe]
            
            Returns:
            None -> printing summary statistics for the numerical and categorical features to the console

        """
        print("Summary statistics for numerical features")
        print(df.describe())
        print("\nSummary statistics for categorical features")
        print(df.describe(include=['O']))



class DataInspector:
    def __init__(self,strategy:DataInspectionStrategy) -> None:
        """
            initialize data inspector with a specific strategy

            parameters:
                strategy -> strategy to be used for data inspection

            Returns:
                None
        """
        self._strategy = strategy
    
    def set_strategy(self,strategy:DataInspectionStrategy):
        """
            setting a new strategy for data inspector

            parameters:
                strategy -> strategy to be used for data inspection

            Returns:
                None
        """
        self._strategy = strategy

    def executing_inspection(self,df:pd.DataFrame):
        """
            executing the inspection using the current strategy

            parameters:
                df -> pd.dataframe [dataframe]
            
            Returns:
                None -> executes the strategy's inspection method
        """
        self._strategy.inspect(df=df)



if __name__ == "__main__":
    #loading the example data
    df = pd.read_csv('D:/coding/ml/mlframework/data/setcalories.csv')

    #initializeing the data inspector with a specific strategy
    inspector = DataInspector(DataTypesInspectionStrategy())
    inspector.executing_inspection(df=df)

    #changing the data inspection strategy
    inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    inspector.executing_inspection(df=df)