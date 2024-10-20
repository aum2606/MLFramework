from abc import ABC,abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def perform_analysis(self, df: pd.DataFrame ,feature:str):
        """
            Perfrom univariate analysis on a specific feature of the dataframe

            Parameters:
                data (pd.DataFrame) -> The dataframe to perform analysis on
                features(str) -> the name of the feature/columns to be analysed
            
            Return:
                None -> this method visualize the distribution of the features
        """


class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def perform_analysis(self, df: pd.DataFrame, feature: str):
        """
            plots the distribution of a numerical feature using a histogram and KDE

            Parameters:
                df -> pd.dataframe [dataframe]
                features -> the name of the features to be analyzed
            
            Return:
                None -> display a histogram with a KDE plot
        """

        plt.figure(figsize=(10,5))
        sns.histplot(df[feature], kde=True,bins=30)
        plt.title(f'distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()


class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def perform_analysis(self, df: pd.DataFrame, feature: str):
        """
            plot the distribution of categorical features using a bar plot

            Parameters:
                df -> pd.dataframe [dataframe]
                features -> the name of the features to be analyzed

            Returns: 
                None -> display a bar plot of the categorical feature
        """

        plt.figure(figsize=(10,5))
        sns.countplot(x=feature, data=df)
        plt.title(f'distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel("count")
        plt.xticks(rotation=45)
        plt.show()



class UnivariateAnalyzer():
    def __init__(self,strategy:UnivariateAnalysisStrategy):
        """
            initializes strategy to be used for univariate analysis

            Parameters:
                strategy(univariateAnalysisStrategy) -> strategy to be used for analysis
        """
        self._strategy=strategy
    
    def set_strategy(self,strategy:UnivariateAnalysisStrategy):
        """
            Set a new strategy for the UnivariateAnalyzer

            Parameters:
                strategy(univariateAnalysisStrategy) -> strategy to be used for analysis

            Returns:
                None
        """
        self._strategy=strategy

    def execute_analysis(self, df: pd.DataFrame, feature:str):
        """
            executes the univariate analysis using the current strategy

            Parameteres:
                df(dataframe) -> dataframe on which analysis is performed
                feature -> feature to be used in analysis

            Returns:
                None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.perform_analysis(df,feature=feature)



if __name__ =='__main__':
    #example
    # df = pd.read_csv('D:/coding/ml/machine learning framework/src/components/extracted_data/AmesHousing.csv')
    # analyser = UnivariateAnalyzer(NumericalUnivariateAnalysis())
    # analyser.execute_analysis(df, 'SalePrice')

    # analyser.set_strategy(CategoricalUnivariateAnalysis())
    # analyser.execute_analysis(df,'Neighborhood')
    pass