from abc import ABC,abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def perform_analysis(self,df: pd.DataFrame,feature1:str , feature2:str):
        """
            Perform bivariate analysis on 2 feature of the dataframe

            Parameters:
                df (pd.DataFrame): The dataframe to perform analysis on
                feature1 (str): The first feature/column
                feature2 (str): the second feature/column

            Returns:
                None -> visualize the relationship between 2 features
        """
        pass



class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def perform_analysis(self, df: pd.DataFrame, feature1: str,feature2:str):
        """
            Plots the relationship between 2 numerical feaures using a scatter plot

            Parameters:
                df(pd.dataframe) -> the data frame to perform analysis on
                feature1 (str): The first feature/column
                feature2 (str): the second feature/column

            Return:
                None -> visualize the relation between 2 features
        """
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=feature1,y=feature2,data=df)
        plt.title(f'Relationship between {feature1} and {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def perform_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
            plots the relationship between one categorical and one numerical feature using box plot

            Parameters:
                df(pd.dataframe) -> the data frame to perform analysis on
                feature1 (str): The name of categorical feature/column
                feature2 (str): the name of numerical feature/column

            Returns:
                None -> diplay a box plot showing relaitonship of the two features
        """
        plt.figure(figsize=(10,6))
        sns.boxplot(x=feature1,y=feature2,data=df)
        plt.title(f'Relationship between {feature1} and {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()


class BivariateAnalyzer():
    def __init__(self,strategy:BivariateAnalysisStrategy):
        """
            initializes strategy to be used for univariate analysis

            Parameters:
                strategy(BivariateAnalysisStrategy) -> strategy to be used for analysis
        """
        self._strategy=strategy
    
    def set_strategy(self,strategy:BivariateAnalysisStrategy):
        """
            Set a new strategy for the UnivariateAnalyzer

            Parameters:
                strategy(BivariateAnalysisStrategy) -> strategy to be used for analysis

            Returns:
                None
        """
        self._strategy=strategy

    def execute_analysis(self, df: pd.DataFrame, feature1:str, feature2:str):
        """
            executes the univariate analysis using the current strategy

            Parameteres:
                df(dataframe) -> dataframe on which analysis is performed
                feature1 -> the name of first feature/column 
                feature2 -> the name of second feature/column 

            Returns:
                None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.perform_analysis(df,feature1=feature1,feature2=feature2)


if __name__ =='__main__':
    #example
    # df = pd.read_csv('D:/coding/ml/machine learning framework/src/components/extracted_data/AmesHousing.csv')
    # analyser = BivariateAnalyzer(NumericalVsNumericalAnalysis())
    # analyser.execute_analysis(df, 'Gr Liv Area' ,'SalePrice')

    # analyser.set_strategy(CategoricalVsNumericalAnalysis())
    # analyser.execute_analysis(df,'Overall Qual','SalePrice')
    pass