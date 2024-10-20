from abc import ABC,abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class MultivariateAnaylsisTemplate(ABC):
    def perform_analysis(self,df : pd.DataFrame):
        """
            Perform comprehensive multivariate anaylsis by generating a correlation heatmap and pairplot

            Parameters:
                df (pd.DataFrame): The dataframe to perform analysis on


            Returns:
                None -> orchestrate the multivariate anaylsis process
        """
        self.generate_correltation_heatmap(df)
        self.generate_pairplot(df)
        
    @abstractmethod
    def generate_correltation_heatmap(self,df:pd.DataFrame):
        """
            Generate a correlation heatmap for the dataframe

            Parameters:
                df(pd.dataframe) -> the dataframe containing the data to be analysed
            
            Returns:
                None -> this method generate and display correlation heatmap
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        """
            Generate a pairplot for the dataframe

            Parameters:
                df(pd.dataframe) -> the dataframe containing the data to be analysed

            Reurns:
                None -> this method generate and display a pair plot
        """
        pass

class MultivariateAnalysis(MultivariateAnaylsisTemplate):
    def generate_correltation_heatmap(self,df:pd.DataFrame):
        """
            Generate and displays a correlation heatmap for numerical features in the dataframe

            Parameters:
                df(pd.dataframe) -> the dataframe containing the data to be analysed

            Returns:
                None -> this method generate and display correlation heatmap
        """
        plt.figure(figsize=(15,7))
        sns.heatmap(data=df.corr(),annot=True,fmt='.2f',cmap='coolwarm',linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        """
            Generate and display a pairplot for the selected features in the dataframe

            Parameters:
                df(pd.dataframe) -> the dataframe containing the data to be analysed

            Returns:
                None -> this method generate and display a pair plot
        """
        plt.figure(figsize=(10,8))
        sns.pairplot(data=df)
        plt.title("Pair plot of selected features",y=1.02)
        plt.show()


if __name__ =='__main__':
    #example
    # df = pd.read_csv('D:/coding/ml/machine learning framework/src/components/extracted_data/AmesHousing.csv')
    # analyzer = MultivariateAnalysis()
    # selected_features = df[['SalePrice','Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Year Built']]
    # analyzer.perform_analysis(selected_features)
    pass
